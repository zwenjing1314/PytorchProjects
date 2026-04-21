import random
from sympy.core import N
import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import functools

matplotlib.use("TkAgg")


# 为再现性设置随机seem
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 如果你想要新的结果就是要这段代码

print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 数据集根目录
dataroot = "d08_pix2pix为黑白图片上色/data/facades"

# 数据加载器并发数
# macOS/Windows 下 DataLoader 多进程会用 spawn 启动子进程。
# 当前脚本的训练代码在文件顶层执行，worker 重新导入脚本时容易异常退出；
# 先使用单进程加载，避免 "DataLoader worker exited unexpectedly"。
workers = 0

# 训练批的大小
batch_size = 32

# 训练图片的大小. 所有图片都会转成改大小的图片
image_size = 256


# 训练图片的通道数。彩色图片是RGB三个通道
input_nc = 3
output_nc = 3

# 生成器的特征图大小
ngf = 64

# 判别器的特征图大小
ndf = 64

# 数据集的训练次数
num_epochs = 200

# 学习率
lr = 0.0002

# Adam优化器的beta1参数
beta1 = 0.5

# 所使用的gpu数量。0表示使用cpu
ngpu = 1

# L1正则化项的系数
lambda_L1 = 100


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# 读取目录下的所有文件名作为数据集
def make_dataset(_dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(_dir), '%s is not a valid directory' % _dir
    # assert os.path.isdir(_dir), f"{_dir} is not a valid directory"

    for root, _, fnames in sorted(os.walk(_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

# 生成图片的处理器
def gen_transform(grayscale=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    transform_list.append(transforms.Resize(image_size))
    transform_list.append(transforms.CenterCrop(image_size))

    # 随机翻转图像，进行数据增强
    transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())

    if grayscale:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    else:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)


class AlignedDataset(torch.utils.data.Dataset):  # 继承了 torch.utils.data.Dataset
    def __init__(self, root, left_is_A=True, phase='train', max_dataset_size=float('inf')):
        self.dir_AB = os.path.join(root, phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, max_dataset_size))  # get image paths
        
        # 生成A和B的图片处理器
        self.A_transform = gen_transform(grayscale=(input_nc == 1))
        self.B_transform = gen_transform(grayscale=(output_nc == 1))
        self.left_is_A = left_is_A

    def __getitem__(self, index):
        # 从文件中加载出图片
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # 切分图片的A/B面
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        if not self.left_is_A:
            A, B = B, A

        A = self.A_transform(A)
        B = self.B_transform(B)

        return A, B

    def __len__(self):
        return len(self.AB_paths)


# 将AB 张量拼接成一张图片
def make_aligned_img(A, B, normA=True):
    A, B = A.cpu(), B.cpu()
    if A.dim() == 3:
        A = A[None, :, :, :]
    if B.dim() == 3:
        B = B[None, :, :, :]

    aligned = torch.cat((B, A), dim=3)
    aligned = vutils.make_grid(aligned, padding=2, normalize=True)
    aligned = np.transpose(aligned, (1, 2, 0))
    return aligned

# 创建数据集
dataset = AlignedDataset(dataroot, left_is_A=False)
"""
  为什么dataset可以传入 DataLoader？

  因为它继承了：
  torch.utils.data.Dataset
  并且实现了两个关键方法：
  def __getitem__(self, index):
      ...
  和：
  def __len__(self):
      ...
"""
# 创建加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# 选择我们运行在上面的设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 打印训练集中的第一个数据
A, B = dataset[0]
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(make_aligned_img(A, B))
plt.show

# 自定义的权重初始化函数，用于初始化netG和netD网络
def weights_init(m):
    """
    卷积：权重从接近 0 的小随机值起步，避免一开始就过大导致梯度爆炸，同时打破对称性（每层神经元不完全相同）。
        0.02 这类取值来自 DCGAN / pix2pix 一脉的做法，针对 GAN 训练比较常用。
    BatchNorm：γ≈1、β=0，相当于一开始 尽量不缩放、不平移，让前面的卷积主导，训练过程中再慢慢学 BN 的校正。
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

"""基于 Unet 生成器"""
class UnetGenerator(nn.Module):
    def __init__(self,input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        参数:
            input_nc (int)  -- 输入图片的通道数
            output_nc (int) -- 输出图片的通道数
            num_downs (int) -- UNet下采样次数. 例如, # 如果 |num_downs| == 7,
                                图片的大小如果是128x128，经过7次下采样后会变成1x1
            ngf (int)       -- 特征图大小
            norm_layer      -- 归一化层

        我们从最里面的层开始逐步往外构建，
        这可以看成是一个递归操作
        """
        super(UnetGenerator, self).__init__()
        # 构建UNet架构
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # 最内层
        for i in range(num_downs - 5):  # 使用 ngf * 8 个filters添加中间层
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # 主键将filters数量从 ngf * 8 降到 ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # 最外层

    def forward(self, input):
        return self.model(input)


"""UNet的子模块，带有残差连接"""
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构建一个带残差连接的UNet子模块

        Parameters:
            outer_nc (int) -- 外层filters数量
            inner_nc (int) -- 内层filters数量
            input_nc (int) -- 输入图片/特征图的通道数
            submodule (UnetSkipConnectionBlock) -- 中间夹的子模块
            outermost (bool)    -- 是否为最外层
            innermost (bool)    -- 是否为最内层
            norm_layer          -- 归一化层
            use_dropout (bool)  -- 是否使用Dropout层
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:  # 判断是否“包装过的归一化层”
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            # 最外层不用加残差连接了
            return self.model(x)
        else:   # 添加残差连接
            return torch.cat([x, self.model(x)], 1)


# 初始化UNet-256。如果是UNet-128，则下采样次数为7
# facade数据集，输入输出都是RGB图像
netG = UnetGenerator(input_nc, output_nc, 8, ngf).to(device)

# 使用多个gpu处理
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 使用自定义的权重初始化函数
netG.apply(weights_init)

# 打印模型
print(netG)


"""构建PatchGAN判别器   n_layers=3 表示 70x70的PatchGAN"""
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        # 判断 norm_layer 传进来的是“普通类”，还是 functools.partial 包装后的对象
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        # 卷积核大小
        kw = 4
        # padding 大小
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        
        # nf_mult 表示当前通道倍率
        # ndf=64 时，nf_mult=1 表示 64 通道
        nf_mult = 1
        # nf_mult_prev 表示上一层的通道倍率
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 逐渐增加filters数量
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

netD = NLayerDiscriminator(input_nc+output_nc, ndf, 3)

# 创建判别器
netD = NLayerDiscriminator(input_nc+output_nc, ndf, 3).to(device)

# 使用多个gpu处理
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 应用weights_init函数随机初始化所有权重，mean= 0，stdev = 0.2
netD.apply(weights_init)

# 打印模型
print(netD)


# 初始化BCELoss函数
criterion = nn.BCEWithLogitsLoss()

# 生成器的L1正则化项
criterionL1 = torch.nn.L1Loss()

# 定义训练期间真假图像的标签值
real_label = torch.tensor(1, dtype=torch.float, device=device)
fake_label = torch.tensor(0, dtype=torch.float, device=device)

# 为 G 和 D 创建 Adam 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


def set_requires_grad(nets, requires_grad=False):
    """将网络的所参数设置 requies_grad=Fasle以减少不必要的计算
    参数:
        nets (network list)   -- 网络列表
        requires_grad (bool)  -- 是否需要计算梯度
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def make_label(prediction, target_is_real):
    if target_is_real:
        target_tensor = real_label
    else:
        target_tensor = fake_label
    return target_tensor.expand_as(prediction)


# 记录训练的过程
img_list = []
G_losses = []
D_losses = []
iters = 0

# 取第一张图为固定的约束，以观察生成器的学习过程
fixed_A, fixed_B = None, None

netG.train()
netD.train()

print("Starting Training Loop...")
# 遍历数据集
for epoch in range(num_epochs):
    # 遍历真实数据中的每一个数据批
    for i, (real_A, real_B) in enumerate(dataloader, 0):
        
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        if fixed_A is None:
            fixed_A = real_A[:1,:,:,:]
            fixed_B = real_B[:1,:,:,:]

        ############################
        # (1) 更新判别器: 最大化 log(D(x)) + log(1 - D(G(z)))
        ###########################

        set_requires_grad(netD, True)  # 需要计算D的梯度
        netD.zero_grad()  # 清空D的梯度

        ## 使用真实数据构建训练批
        # 拼接图像和约束
        real_AB = torch.cat((real_A, real_B), 1)
        # 预测真实数据
        pred_real = netD(real_AB)
        # 制作用于计算损失的label
        label = make_label(pred_real, True)
        # 计算真实数据的损失
        errD_real = criterion(pred_real, label)


        ## 使用生成器产生的数据构建训练批
        # 使用生成器产生假数据
        fake_B = netG(real_A)
        # 拼接 假图和约束
        fake_AB = torch.cat((real_A, fake_B), 1)
        # 通过将fake_B从计算图中剥离来隔断梯度传播到G
        pred_fake = netD(fake_AB.detach())
        # 制作用于计算损失的label
        label = make_label(pred_fake, False)
        # 计算假数据的损失
        errD_fake = criterion(pred_fake, label)

        # 合并损失，计算梯度
        errD = (errD_real + errD_fake) * 0.5
        errD.backward()

        # 更新判别器
        optimizerD.step()

        ############################
        # (2) 更新生成器: 最大化 log(D(G(z)))
        ###########################
        set_requires_grad(netD, False)  # 不需要 计算D的梯度
        netG.zero_grad()  # 清空生成器的梯度

        # 首先生成假图
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB)
        # 制作用于计算损失的label，并计算损失
        label = make_label(pred_fake, True)  # 假数据在生成器看来是真数据
        errG_GAN = criterion(pred_fake, label)

        # 合并损失，计算梯度
        errG_L1 = criterionL1(fake_B, real_B) * lambda_L1
        errG = errG_GAN + errG_L1

        errG.backward()
        # 更新生成器
        optimizerG.step()

        # 保存训练时的损失，用于后续打印图形
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 记录生成器将fixed_noise映射成了什么样的图片
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_A).detach().cpu()
            #  img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            img_list.append(fake)

        iters += 1
    
    # 输出训练状态
    print('[%3d/%3d] D_fake: %.4f  D_real: %.4f  G_GAN: %.4f  G_L1: %.4f'
          % (epoch+1, num_epochs,
             errD_fake.item(), errD_real.item(), errG_GAN.item(), errG_L1.item()))


# 保存训练后的模型参数到当前脚本目录；在 notebook 中则退回到当前工作目录
save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
netG_to_save = netG.module if isinstance(netG, nn.DataParallel) else netG
netD_to_save = netD.module if isinstance(netD, nn.DataParallel) else netD

netG_path = os.path.join(save_dir, 'netG_pix2pix.pt')
netD_path = os.path.join(save_dir, 'netD_pix2pix.pt')

torch.save(netG_to_save.state_dict(), netG_path)
torch.save(netD_to_save.state_dict(), netD_path)

print(f"Saved generator weights to: {netG_path}")
print(f"Saved discriminator weights to: {netD_path}")

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


##%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(make_aligned_img(fixed_A, i), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())



def forwad_next():
    # netG.load_state_dict(torch.load('netG_pix2pix.pt', map_location=device))
    netG.eval()

    real_A, real_B = next(iter(dataloader))
    
    real_A, real_B = real_A.to(device), real_B.to(device)
    with torch.no_grad():
        fake_B = netG(real_A)
        
    return real_A.cpu(), real_B.cpu(), fake_B.detach().cpu()


# 生成假图
real_A, real_B, fake_B = forwad_next()
pairs = torch.cat((real_A, real_B, fake_B), dim=2)
pairs_img = np.transpose(vutils.make_grid(pairs[:5], padding=2, normalize=True).cpu(),(1,2,0))

# 打印真实图片
plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Real/Fake Images")
plt.imshow(pairs_img)

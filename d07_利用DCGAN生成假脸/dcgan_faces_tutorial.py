import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

matplotlib.use('TkAgg')
# 为再现性设置随机seed
manualSeed = 42

print("Random seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 数据集根目录
dataroot = "data/celeba"

# 数据加载器并发数
workers = 2

# 训练批的大小
batch_size = 128

# 训练图片的大小. 所有图片都会转成改大小的图片
image_size = 64

# 训练图片的通道数。彩色图片是RGB三个通道
nc = 3

# 噪声向量大小（生成器的输入大小）
nz = 100

# 生成器的特征图大小
ngf = 64

# 判别器的特征图大小
ndf = 64

# 数据集的训练次数
num_epochs = 5

# 学习率
lr = 0.0002

# Adam优化器的beta1参数
beta1 = 0.5

# 所使用的gpu数量。0表示使用cpu
ngpu = 1

# 我们可以按照设置的方式使用图像文件夹数据集。
# 创建数据集
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
# 创建加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 选择我们运行在上面的设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 绘制部分我们的输入图像
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()


# 自定义的权重初始化函数，用于初始化netG和netD网络
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 生成器代码
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是Z，输入到卷积层
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# 实例化一个生成器
netG = Generator(ngpu).to(device)

# 使用多个gpu处理
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 使用自定义的权重初始化函数
netG.apply(weights_init)

# 打印模型
print(netG)


# 判别器代码
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# 创建判别器
netD = Discriminator(ngpu).to(device)

# 使用多个gpu处理
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 应用weights_init函数随机初始化所有权重，mean= 0，stdev = 0.2
netD.apply(weights_init)

# 打印模型
print(netD)

# 初始化BCELoss函数
criterion = nn.BCELoss()

# 创建一批固定的噪声数据，我们将用它来可视化生成器的学习进程
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 定义训练期间真假图像的标签值
real_label = 1
fake_label = 0

# 为 G 和 D 创建 Adam 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# DCGAN的训练过程

# 记录训练的过程
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# 遍历数据集
for epoch in range(num_epochs):
    # 遍历真实数据中的每一个数据批
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) 更新判别器: 最大化 log(D(x)) + log(1 - D(G(z)))
        ###########################
        # 使用真实数据构建训练批
        netD.zero_grad()
        # 格式化训练批
        """
        data: 从 dataloader 获取的批次数据，类型为 list [images, labels]
        data[0]: 图像张量
        real_cpu shape: (128, 3, 64, 64)
        128 张图片，每张 3 通道，64×64 像素
        """
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)  # b_size: 128（批次大小）
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)  # 一维张量，全部填充为 1.0（真实标签）
        # 前向传播真实数据的训练批到
        output = netD(real_cpu).view(-1)  # 输入形状 (128, 3, 64, 64)， 展平 变为一维张量
        # 计算真实数据的损失
        errD_real = criterion(output, label)
        # 反向传播计算梯度
        errD_real.backward()
        D_x = output.mean().item()

        # 使用生成器产生的数据构建训练批
        # 产生正态分布的噪声
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 使用生成器产生假数据
        fake = netG(noise)
        label.fill_(fake_label)
        #为假数据
        output = netD(fake.detach()).view(-1)
        # 计算假数据训练批的损失
        errD_fake = criterion(output, label)
        # 计算梯度
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 将真实数据和假数据产生的梯度加起来
        errD = errD_real + errD_fake
        # 更新判别器
        optimizerD.step()

        ############################
        # (2) 更新生成器: 最大化 log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 假数据在生成器看来是真数据
        # 因为判别器在(1)中更新了，所以我们重新进行一次前向传播
        output = netD(fake).view(-1)
        # 计算损失
        errG = criterion(output, label)
        # 计算G的梯度
        errG.backward()
        D_G_z2 = output.mean().item()
        # 更新生成器
        optimizerG.step()

        # 输出训练状态
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 保存训练时的损失，用于后续打印图形
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 记录生成器将fixed_noise映射成了什么样的图片
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
 
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

real_batch = next(iter(dataloader))

# 打印真实图片
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# 打印假图片
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
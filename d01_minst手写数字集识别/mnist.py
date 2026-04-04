import matplotlib

matplotlib.use("TkAgg")  # 一定要放在 import pyplot 前面，避开 PyCharm 的 interagg 后端

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from utils import SimpleMNIST

"""
torchvision 是 PyTorch 的官方计算机视觉扩展库，主要提供：
1. 数据集 (torchvision.datasets)
    datasets.MNIST(...)  # MNIST 手写数字集
    datasets.CIFAR10(...)  # CIFAR-10 图像分类数据集
    datasets.ImageFolder(...)  # 自定义图像文件夹数据集
2. 模型 (torchvision.models)
    预训练的深度学习模型
    resnet50(pretrained=True)
    vgg16(pretrained=True)
    mobilenet_v2(pretrained=True)
3. 图像变换 (torchvision.transforms)
    数据增强和预处理：
    transforms.ToTensor()      # PIL 图片转 Tensor
    transforms.Normalize()     # 归一化
    transforms.RandomCrop()    # 随机裁剪
    transforms.Resize()        # 缩放
4. 工具函数 (torchvision.utils)
    make_grid()    # 拼接图像网格
    save_image()   # 保存图片
简单说：PyTorch 负责神经网络，torchvision 负责图像处理相关的辅助功能，两者配合使用
"""
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt

# 1. 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 2. 数据集
"""
1. 归一化计算过程
normalized_pixel = (pixel - mean) / std
2. 为什么要进行归一化？
原始图片像素值范围：[0, 1]（ToTensor() 之后）
           ↓
减去均值 0.1307：
   - 原来 0 的像素 → -0.1307
   - 原来 1 的像素 → 0.8693
   - 数据分布中心从 0.5 移到接近 0
           ↓
除以标准差 0.3081：
   - 数据被缩放到标准正态分布 N(0, 1)
   - 大部分数据落在 [-1, 1] 区间
           ↓
输入到神经网络
"""
transform = transforms.Compose([  # TODO: 注意 这里面是一个列表，不能在这里面进行多行注释
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root="./datas",  # 数据存储的根目录
        train=True,  # True=训练集 (60000 张)，False=测试集 (10000 张)
        download=True,  # 如果数据不存在，自动下载
        transform=transform  # 对每张应用的预处理操作
    ),
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root="./datas",
        train=False,
        download=True,
        transform=transform
    ),
    batch_size=64,
    shuffle=False,
    num_workers=0
)


# 3. 显示图片
def imshow(img):
    # img: (C, H, W)
    img = img * 0.3081 + 0.1307  # 按 MNIST 的 Normalize 参数做反归一化
    img = img.clamp(0, 1)  # 防止超出显示范围
    nping = img.cpu().numpy()

    plt.figure(figsize=(8, 4))
    if nping.shape[0] == 1:  # 黑白图单通道
        plt.imshow(nping.squeeze(0), cmap="gray")
    else:
        plt.imshow(np.transpose(nping, (1, 2, 0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# 取一个 batch 看看
dataiter = iter(train_loader)
images, labels = next(dataiter)


# 把 64 张图片（每张图片是一个张量）拼接成一张大的网格图。nrow=8 表示每行放 8 张图片
# 返回一个形状为 (C, H, W) 的大图像张量，可以直接用 matplotlib 显示
# 例如：64 张图片，每行 8 张，会自动排成约 8 行
# imshow(torchvision.utils.make_grid(images[:64], nrow=8))


# 4. 定义网络
class Net(nn.Module):  # LeNet-5 网络架构
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 28x28 -> 24x24
        self.conv2 = nn.Conv2d(6, 16, 5)  # 12x12 -> 8x8
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 24x24 -> 12x12
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 8x8 -> 4x4
        x = torch.flatten(x, 1)  # 比 view 更直观
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)
print(net)
print(type(net))  # <class '__main__.Net'>

# 5. 前向传播 / 损失
sample_images = images[:2].to(device)
sample_labels = labels[:2].to(device)

out = net(sample_images)
print("sample_images.shape:", sample_images.shape)
print("sample_labels:", sample_labels)
print("out:", out)

# 交叉熵损失计算
"""
# 对第 1 张图片的 out[0] 做 softmax:
原始得分：[2.1, -0.5, 3.2, 1.0, -1.2, 0.8, 2.5, -0.3, 1.8, 0.2]

计算指数：
e^2.1 ≈ 8.17
e^(-0.5) ≈ 0.61
e^3.2 ≈ 24.53
...

求和：Σ = 8.17 + 0.61 + 24.53 + ... ≈ 65.42

归一化：
prob[0] = 8.17 / 65.42 ≈ 0.125  # 预测为 0 的概率
prob[1] = 0.61 / 65.42 ≈ 0.009  # 预测为 1 的概率
prob[2] = 24.53 / 65.42 ≈ 0.375 # 预测为 2 的概率
...

softmax 结果：
[0.125, 0.009, 0.375, 0.042, 0.005, 0.032, 0.183, 0.011, 0.091, 0.027]
# 所有概率相加 = 1.0

"""
criterion = nn.CrossEntropyLoss()
loss = criterion(out, sample_labels)
print("sample loss:", loss.item())

# 6. 优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)


# 7. 训练
def train(epoch):
    # 设置网络为训练模式（启用 Dropout、BatchNorm 等层的训练行为）
    net.train()
    # 累计损失值
    running_loss = 0.0

    # enumerate(train_loader, start=1) 从 1 开始计数
    # i: 当前是第几个 batch（从 1 开始）
    # inputs: 一个 batch 的输入图像，形状 (64, 1, 28, 28)
    # labels: 对应的标签，形状 (64,)
    for i, (inputs, labels) in enumerate(train_loader, start=1):
        # 将数据移到设备（GPU 或 CPU）上
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度（PyTorch 默认会累加梯度，所以每次迭代前必须清零）
        optimizer.zero_grad()

        # 前向传播：将输入送入网络，得到输出
        # outputs 形状：(64, 10)，表示 64 张图片属于 10 个数字类别的得分
        outputs = net(inputs)

        # 计算损失：交叉熵损失函数，衡量预测值与真实标签的差距
        # criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)

        # 反向传播：根据损失计算每个参数的梯度
        # 自动微分，从 loss 开始向后传播，计算 ∂loss/∂w 和 ∂loss/∂b
        loss.backward()

        # 更新参数：优化器根据梯度更新网络权重
        # optimizer = optim.SGD(..., lr=0.01, momentum=0.9)
        optimizer.step()

        # 累加当前 batch 的损失值（.item() 取出张量中的标量数值）
        running_loss += loss.item()
        # 每 100 个 batch 打印一次平均损失
        if i % 100 == 0:
            print(f"[Epoch {epoch}] step {i:4d}, loss: {running_loss / 100:.4f}")
            running_loss = 0.0


# 跑 1 轮
train(1)

# 8. 测试
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100.0 * correct / total:.2f}%")
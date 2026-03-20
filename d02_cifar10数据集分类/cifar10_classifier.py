import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")  # 一定要放在 import pyplot 前面，避开 PyCharm 的 interagg 后端

# torch.nn（Neural Network）是 PyTorch 中构建神经网络的核心模块，它提供了构建和训练深度学习模型所需的所有基础组件。
import torch.nn as nn
# torch.nn.functional（通常简写为 F）是 PyTorch 中的函数式神经网络接口，提供了各种无状态的神经网络操作函数。
import torch.nn.functional as F

# torch.optim（Optimizer）是 PyTorch 中的优化器模块，负责实现各种优化算法，用于更新神经网络的参数，使损失函数最小化。让我结合你的代码详细说明：
import torch.optim as optim

"""
原始文件 (cifar-10-python.tar.gz)
    ↓ 解压
pickle 二进制文件 (data_batch_1~5, test_batch)
    ↓ 读取
字典格式 {'data': (10000, 3072), 'labels': (10000,)}
    ↓ reshape & transpose
未归一化图像 (10000, 3, 32, 32), 范围 [0, 255]
    ↓ ToTensor()
归一化到 [0,1] (10000, 3, 32, 32), torch.float32
    ↓ Normalize(mean=[0.5]*3, std=[0.5]*3)
标准化到 [-1,1] (10000, 3, 32, 32), 均值≈0, 标准差≈1
    ↓ DataLoader
批量数据 (batch_size=4, 3, 32, 32)
    ↓ 输入网络
Conv1 层输入：(4, 3, 32, 32)
"""
transform = transforms.Compose(  # 数据变换操作
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)  # 图片像素值范围 [0, 1] -> [-1, 1]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # PyTorch 格式 (C, H, W) -> matplotlib 格式 (H, W, C)
    plt.show()


dataiter = iter(trainloader)
images, labels = next(dataiter)


# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):  # nn.Module 所有神经网络层的父类，提供参数管理、设备迁移、序列化等核心功能。
    """
    1. 继承 nn.Module 后，Net 类自动获得以下能力：
       ├── 参数自动注册：self.conv1 等层的权重自动加入参数列表
       ├── .parameters() 方法：获取所有可训练参数
       ├── .to(device) 方法：在 CPU/GPU 间迁移
       ├── .train() / .eval() 模式切换
       └── state_dict() 模型序列化

    2. 当你执行 net = Net() 时：
       └─> 自动遍历所有 nn.Module 子类属性
           └─> 将 conv1, conv2, fc1 等的参数注册到 net 的参数列表中
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # nn.Conv2d 实现 2D 卷积操作，用于提取图像的空间特征。
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 初始输入：RGB 图片 (3, 32, 32)
        # Conv1: 3→6, 5×5, 无 padding
        # 输出尺寸：(32-5+1) = 28
        """
        conv1 = nn.Conv2d(3, 6, 5) 的内部结构：
        ├── 权重 weight: (6, 3, 5, 5)  # [输出通道，输入通道，高，宽]
        │   └─> 共 6×3×5×5 = 450 个可学习参数
        └── 偏置 bias: (6,)  # 每个输出通道一个偏置
            └─> 共 6 个可学习参数

        前向传播过程（以 batch_size=4 为例）：
        输入：x shape = (4, 3, 32, 32)  # [批量，通道，高，宽]
                 ↓
        卷积运算：
        对于每个输出通道 out_channel ∈ {0,1,2,3,4,5}:
            output[:, out_channel, :, :] =
                Σ(in_channel=0 to 2) conv2d(
                    input[:, in_channel, :, :],
                    weight[out_channel, in_channel, :, :]
                ) + bias[out_channel]
                 ↓
        中间结果：shape = (4, 6, 28, 28)  # 32-5+1=28
                 ↓
        ReLU 激活：F.relu() 逐元素取 max(0, x)
                 ↓
        池化：self.pool(MaxPool2d(2,2))
            └─> 对每个 2×2 区域取最大值
            └─> 输出：(4, 6, 14, 14)
        """
        x = self.pool(F.relu(self.conv1(x)))  # (4, 6, 28, 28) -> (4, 6, 14, 14)
        # Conv2: 6→16, 5×5
        # 输出尺寸：(14-5+1) = 10
        x = self.pool(F.relu(self.conv2(x)))  # (4, 16, 10, 10) -> (4, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)  # 展平：把空间维度 (H, W) 压缩掉 (4, 400)
        x = F.relu(self.fc1(x))  # （4, 120)
        x = F.relu(self.fc2(x))  # （4, 84)
        x = self.fc3(x)  # （4, 10)
        return x  # (批次大小， 分类结果）


net = Net()

criterion = nn.CrossEntropyLoss()
"""
神经网络训练的本质：
找到一组最优参数 θ*（权重和偏置），使得损失函数 L(θ) 最小

数学表达：
θ* = argmin L(θ)

其中：
- θ = {conv1.weight, conv1.bias, conv2.weight, ..., fc3.bias}
- L(θ) = 交叉熵损失（预测值与真实值的差异）

优化器的任务：
通过迭代的方式，不断更新 θ，让 L(θ) 越来越小
"""
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")

dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

ooutputs = net(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")  # 一定要放在 import pyplot 前面，避开 PyCharm 的 interagg 后端

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

transform = transforms.Compose(
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 初始输入：RGB 图片 (3, 32, 32)
        # Conv1: 3→6, 5×5, 无 padding
        # 输出尺寸：(32-5+1) = 28
        x = self.pool(F.relu(self.conv1(x)))  # (4, 6, 28, 28) -> (4, 6, 14, 14)
        # Conv2: 6→16, 5×5
        # 输出尺寸：(14-5+1) = 10
        x = self.pool(F.relu(self.conv2(x)))  # (4, 16, 10, 10) -> (4, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)  # 展平：把空间维度 (H, W) 压缩掉 (4, 400)
        x = F.relu(self.fc1(x))  # （4, 120)
        x = F.relu(self.fc2(x))  # （4, 84)
        x = self.fc3(x)  # （4, 10)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
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
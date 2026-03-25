from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")  # 一定要放在 import pyplot 前面，避开 PyCharm 的 interagg 后端
import time
import os
import copy

cudnn.benchmark = True
plt.ion()  # interactive mode

# 利用了字典将 train 和 val 同时进行操作
# 在训练集上：扩充、归一化
# 在验证集上：归一化
data_transforms = {  # 定义一个数据转换器
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print(image_datasets)
# print(dataloaders)

# print(dataloaders['train'])
# print(iter(dataloaders['train']))
# print(dataloaders['train'])
# next(iter(dataloaders['train']))
# next(iter(dataloaders['train']))
# print(iter(dataloaders['train']))
# next(iter(dataloaders['train']))
# next(iter(dataloaders['train']))
# next(iter(dataloaders['train']))
# print(dataloaders['train'])
# print(iter(dataloaders['train']))


# print(next(iter(dataloaders['train'])))
# print(dataset_sizes)
# print(class_names)


def imshow(inp, title=None):
    # 可视化一组 Tensor 的图片
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # 将图片范围限制在 [0, 1]
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(4)  # 暂停一会儿，为了将图片显示出来


# 获取一批训练数据
inputs, classes = next(iter(dataloaders['train']))
# 批量制作网格
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """ 训练模型，并返回在验证集上的最佳模型和准确率
    Args:
    - model(nn.Module): 要训练的模型
    - criterion: 损失函数
    - optimizer(optim.Optimizer): 优化器
    - scheduler: 学习率调度器
    - num_epochs(int): 最大 epoch 数
    Return:
    - model(nn.Module): 最佳模型
    - best_acc(float): 最佳准确率
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 训练集和验证集交替进行前向传播
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式，可以更新网络参数
            else:
                model.eval()  # 设置为预估模式，不可更新网络参数

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据集
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清空梯度，避免累加了上一次的梯度
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # 正向传播
                    outputs = model(inputs)
                    """
                    torch.max(outputs, 1) 
                    沿着维度1（即特征维度）寻找最大值
                    返回两个值：最大值本身和对应的索引
                    使用 _ 忽略最大值，只保留索引（预测的类别）
                    preds 变量存储了模型对每个输入样本的预测类别
                    在分类任务中，outputs 通常是模型最后一层的输出，形状为 [batch_size, num_classes]，其中每一行表示一个样本在各个
                    类别上的预测分数。通过 torch.max(outputs, 1)，我们找到每个样本预测分数最高的类别作为最终预测结果。
                    """
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播且仅在训练阶段进行优化
                    if phase == 'train':
                        loss.backward()  # 反向传播
                        optimizer.step()

                # 统计loss、准确率
                running_loss += loss.item() * inputs.size(0)  # 平均损失 * 批次大小
                running_corrects += torch.sum(preds == labels.data)

            """"
            scheduler.step() 的主要功能是：
                每个 epoch 结束后自动调整学习率
                根据预定义的调度策略逐渐降低（或改变）学习率
                帮助模型在训练后期更精细地收敛到最优解
                
            完整的数据流向
                初始化 optimizer (lr=0.001)
                    ↓
                初始化 scheduler (记录当前 lr)
                    ↓
                开始 Epoch 0
                    ↓
                训练多个批次 (使用当前 lr=0.001)
                    ↓
                Epoch 结束 → scheduler.step()
                    ↓
                scheduler 根据策略计算新 lr
                    ↓
                更新 optimizer.param_groups[0]['lr']
                    ↓
                开始 Epoch 1 (使用新的 lr)
            """
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 发现了更优的模型，记录起来
            if phase == 'val' and epoch_acc > best_acc:  # 监督过拟合
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载训练的最好的模型
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    """
    可视化模型的预测结果

    参数:
        model: 训练好的 PyTorch 模型
        num_images: 要显示的图片数量，默认为 6 张
    """
    # ========== 步骤 1: 保存并设置模型模式 ==========
    was_training = model.training  # 保存当前训练状态（True/False）
    # eval() 的作用：
    # - 关闭 Dropout 层（不再随机丢弃神经元）
    # - 固定 BatchNorm 层的统计量（使用训练好的均值和方差）
    # 确保预测结果稳定一致
    model.eval()  # 切换到评估模式
    images_so_far = 0   # 已显示的图片计数器
    fig = plt.figure()   # 创建一个新的图形窗口

    # ========== 步骤 2: 禁用梯度计算 ==========
    with torch.no_grad():   # 上下文管理器，不计算梯度
        # 好处：
        # 1. 节省内存（不需要存储计算图）
        # 2. 加速计算（跳过梯度相关的运算）
        # 3. 评估时本来就不需要反向传播

        # ========== 步骤 3: 遍历验证集数据 ==========
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            # inputs shape: (batch_size, 3, 224, 224)
            # labels shape: (batch_size,)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # torch.max 返回两个值：(最大值，最大值的索引)
            # _ : 最大值（我们不需要）
            # preds: 最大值的索引 → 预测的类别标签
            # dim=1 表示在类别维度上找最大值
            # 示例：
            # outputs = [[0.3, 0.7],    → preds = [1, 0]
            #            [0.8, 0.2]]       第 1 张预测为类别 1，第 2 张预测为类别 0
            _, preds = torch.max(outputs, 1)

            # ========== 步骤 5: 逐张图片可视化 ==========
            for j in range(inputs.size()[0]):  # 遍历 batch 中的每张图片
                # inputs.size()[0] = batch_size
                # 例如：如果 batch_size=4，则 j=0,1,2,3
                images_so_far += 1

                # 创建子图并显示图片
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')

                # 设置标题显示预测结果
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                # 显示图片
                imshow(inputs.cpu().data[j])

                # ========== 步骤 6: 检查是否达到目标数量 ==========
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

            # 如果当前 batch 的图片不够，继续下一个 batch
            # （一个 batch 可能只有 4 张，但我们需要显示 6 张）

        # ========== 步骤 7: 遍历完所有数据后的清理 ==========
        model.train(mode=was_training)
        # 如果之前是训练模式，现在切回训练模式
        # 这样后续的训练代码不会受影响


model = models.resnet18(pretrained=True)  # 加载预训练模型
num_ftrs = model.fc.in_features  # 获取低级特征维度
model.fc = nn.Linear(num_ftrs, 2)  # 替换新的输出层
model = model.to(device)
# 交叉熵作为损失函数
criterion = nn.CrossEntropyLoss()
# 所有参数都参加训练
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 每过 7 个 epoch 将学习率变为原来的 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model, criterion, optimizer_ft,
                       scheduler, num_epochs=3)  # 开始训练

visualize_model(model_ft)

model_conv = torchvision.models.resnet18(pretrained=True) # 加载预训练模型
for param in model_conv.parameters(): # 锁定模型所有参数
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features  # 获取低级特征维度
model_conv.fc = nn.Linear(num_ftrs, 2)  # 替换新的输出层

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 只有最后一层全连接层fc，参加训练
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# 每过 7 个 epoch 将学习率变为原来的 0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=3)

visualize_model(model_conv)

plt.ioff()
plt.show()
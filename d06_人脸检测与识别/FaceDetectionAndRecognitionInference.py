# mport 非常灵活，可以导入模块、类、函数、变量等几乎所有 Python 对象
from facenet_pytorch import MTCNN, InceptionResnetV1  # 两个模型定义的类
import torch
from torch.utils.data import DataLoader  # 加载数据的模块 （模块是有别与类和函数的）
from torchvision import datasets  # 处理图像的模块
import pandas as pd
import os

# 设置数据加载的工作进程数：Windows 系统设为 0（不支持多进程），其他系统设为 4
workers = 0 if os.name == 'nt' else 4

# 检测设备是否可用 GPU，如果可用则使用 CUDA:0，否则使用 CPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('Running on device: {}'.format(device))

# 初始化 MTCNN 人脸检测模型
# image_size=160: 检测到的人脸会被缩放到 160x160 像素，这是后续识别模型的输入尺寸
# margin=0: 检测框周围不添加额外边距
# min_face_size=20: 最小检测人脸尺寸为 20 像素，小于这个尺寸的人脸将被忽略
# thresholds=[0.6, 0.7, 0.7]: MTCNN 三个阶段的置信度阈值，逐步过滤非人脸区域
# factor=0.709: 图像金字塔缩放因子，用于检测不同尺度的人脸
# post_process=True: 对检测到的人脸进行后处理（对齐和标准化），提升识别准确率
# device=device: 指定模型运行设备（GPU 或 CPU）
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# 初始化 InceptionResnetV1 人脸识别模型
# pretrained='vggface2': 加载在 VGGFace2 大规模人脸数据集上预训练的权重
# pretrained=None 示从本地~/.cache/torch/hub/checkpoints/ 加载权重20180402-114759-vggface2.pt
# .eval(): 将模型设置为评估模式（关闭 Dropout 等训练时的特殊层）
# .to(device): 将模型移动到 GPU 或 CPU 设备上
# 该模型会提取人脸的 512 维特征向量，用于后续的人脸比对和识别
# resnet = InceptionResnetV1(pretrained=None).eval().to(device)
resnet = InceptionResnetV1(pretrained=None)
state_dict = torch.load('/home/wenjing/.cache/torch/hub/checkpoints/20180402-114759-vggface2.pt')
resnet.load_state_dict(state_dict, strict=False)
resnet = resnet.eval().to(device)



# 定义数据整理函数，用于 DataLoader 加载数据
# DataLoader 默认会将多个样本打包成 batch，但这里我们希望对每个样本单独处理
# x 是一个包含单个样本的列表，返回 x[0] 即取出该样本
def collate_fn(x):
    return x[0]


# 使用 ImageFolder 加载测试图像数据集
# ImageFolder 会自动读取 ./datas/test_images 目录下的所有子文件夹作为不同类别
dataset = datasets.ImageFolder('./data')

# 构建反向映射字典：将类别索引映射回类别名称（人名）
# dataset.class_to_idx 是 {'person1': 0, 'person2': 1, ...} 的格式
# 我们需要反转它为 {0: 'person1', 1: 'person2', ...} 方便后续查找
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

# 创建数据加载器
# collate_fn=collate_fn: 使用自定义的整理函数，一次处理一个样本
# num_workers=workers: 使用之前定义的工作进程数来并行加载数据
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

# 初始化两个列表用于存储处理后的人脸图像和对应的人名
aligned = []  # 存储对齐后的人脸张量
names = []  # 存储对应的人名标签
for x, y in loader:  # x: <PIL.Image.Image image mode=RGB size=960x744 at 0x764784A3CB70>  PIL对像 图片的数据
    # 使用 MTCNN 检测图像中的人脸
    # 返回 x_aligned: 对齐后的人脸张量（已自动裁剪并缩放到 160x160）
    # 返回 prob: 检测到人脸的置信度概率
    x_aligned, prob = mtcnn(x, return_prob=True)
    # x_aligned: torch.Size([3, 160, 160])  rgb  height * weight 检测框
    # 如果成功检测到人脸（x_aligned 不为 None）
    if x_aligned is not None:
        # 打印人脸检测的置信度概率
        print('Face detected with probability: {:8f}'.format(prob))
        # 将对齐后的人脸张量添加到列表中
        aligned.append(x_aligned)
        # 通过标签索引 y 查找对应的人名，并添加到列表中
        names.append(dataset.idx_to_class[y])

# 将 aligned 列表中的所有对齐人脸张量堆叠成一个批次的张量
# torch.stack(aligned): 沿着新维度堆叠所有张量，形状从 [(C, H, W), (C, H, W), ...]
#                     变为 (N, C, H, W)，其中 N 是人脸数量
# .to(device): 将张量移动到 GPU（或 CPU）设备上，以便进行后续的模型推理
aligned = torch.stack(aligned).to(device)  # n * 3 * 160 * 160

# 使用 InceptionResnetV1 模型提取人脸特征向量（嵌入）
# resnet(aligned): 将批量人脸图像输入网络，输出 512 维特征向量
#                  每个特征向量表示一个人在高维空间中的"指纹"
# .detach(): 从计算图中分离张量，停止梯度追踪，因为这是推理阶段不需要反向传播
# .cpu(): 将结果从 GPU 移回 CPU，方便后续处理和查看
# embeddings 的形状为 (N, 512)，即 N 个人脸，每个人脸用 512 个浮点数表示
embeddings = resnet(aligned).detach().cpu()  # n * 512


# 计算所有人脸特征向量之间的欧氏距离矩阵
# 外层列表推导式 [for e1 in embeddings]: 遍历每个人脸的特征向量作为基准
# 内层列表推导式 [for e2 in embeddings]: 遍历每个人脸的特征向量作为比较对象
# (e1 - e2).norm(): 计算两个特征向量之间的差值，然后求 L2 范数（欧氏距离）
# .item(): 将标量张量转换为 Python 数值
# dists 是一个 N×N 的二维列表，dists[i][j] 表示第 i 个人和第 j 个人脸特征的相似度距离
# 距离越小表示两个人越相似（可能是同一个人），距离越大表示差异越大
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
# 将距离矩阵转换为 DataFrame 格式并打印
# columns=names: 设置列名为人名列表
# index=names: 设置行索引为人名列表
# 最终输出一个美观的表格，可以直观地看到任意两个人脸之间的相似度距离
# 对角线应该全为 0（自己和自己的距离为 0）
# 同一人的不同照片之间距离应该很小
# 不同人之间的距离应该较大
print(pd.DataFrame(dists, columns=names, index=names))



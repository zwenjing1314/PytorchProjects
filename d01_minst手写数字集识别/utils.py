import torch
from torch.utils.data import Dataset, DataLoader  # 基类/接口
import numpy as np
import struct
import gzip
from PIL import Image

from torchvision import datasets, transforms


class ManualMNIST(Dataset):
    def __init__(self, root='./datas', train=True, transform=None):
        super().__init__()
        self.transform = transform

        # 根据 train 参数选择文件
        if train:
            images_file = f'{root}/MNIST/raw/train-images-idx3-ubyte.gz'
            labels_file = f'{root}/MNIST/raw/train-labels-idx1-ubyte.gz'
        else:
            images_file = f'{root}/MNIST/raw/t10k-images-idx3-ubyte.gz'
            labels_file = f'{root}/MNIST/raw/t10k-labels-idx1-ubyte.gz'

        # 1. 读取并解析图片数据
        with gzip.open(images_file, 'rb') as f:
            # IDX 文件格式头部：
            # - 前 4 字节：魔数 (0x00000803 表示 3D 数组)
            # - 接下来 4 字节：图片数量
            # - 接下来 4 字节：行数 (28)
            # - 接下来 4 字节：列数 (28)
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            # 读取所有像素数据
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

        # 2. 读取并解析标签数据
        with gzip.open(labels_file, 'rb') as f:
            # IDX 标签文件头部：
            # - 前 4 字节：魔数 (0x00000801 表示 1D 数组)
            # - 接下来 4 字节：标签数量
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        # 转换为 PyTorch 张量
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

        print(f"Loaded {len(self.images)} images from {images_file}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # 将 (28, 28) 转为 PIL 图片
        img_pil = Image.fromarray(img.numpy(), mode='L')

        # 应用预处理
        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label


# 使用示例
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 创建数据集
manual_train_dataset = ManualMNIST(root='./datas', train=True, transform=transform)
manual_test_dataset = ManualMNIST(root='./datas', train=False, transform=transform)

# 创建 DataLoader
manual_train_loader = DataLoader(
    manual_train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

manual_test_loader = DataLoader(
    manual_test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# 测试一下
dataiter = iter(manual_train_loader)
images, labels = next(dataiter)
print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")


# 更简单的方法：直接读取 PyTorch 处理好的文件
class SimpleMNIST(Dataset):
    def __init__(self, root='./datas', train=True, transform=None):
        super().__init__()
        self.transform = transform

        if train:
            data_path = f'{root}/MNIST/processed/training.pt'
        else:
            data_path = f'{root}/MNIST/processed/test.pt'

        # 直接加载处理好的张量
        self.data, self.targets = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]

        img_pil = Image.fromarray(img.numpy(), mode='L')

        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label

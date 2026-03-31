import matplotlib.pyplot as plt
from torchvision.io import read_image
import matplotlib
import torch

matplotlib.use('TkAgg')

from pathlib import Path
BASE_PATH = Path(__file__).resolve().parents[1]

img_path = BASE_PATH / "d05_视觉图像检测基于Mask_R-CNN" / "data" / "PennFudanPed" / "PNGImages" / "FudanPed00046.png"
mask_path = BASE_PATH / "d05_视觉图像检测基于Mask_R-CNN" / "data" / "PennFudanPed" / "PedMasks" / "FudanPed00046_mask.png"


image = read_image(str(img_path))
mask = read_image(str(mask_path))

print("=" * 60)
print("图像基本信息")
print("=" * 60)
print(f"原始图像 shape: {image.shape}")
print(f"原始图像 dtype: {image.dtype}")
print(f"原始图像值范围：[{image.min()}, {image.max()}]")

print("\n" + "=" * 60)
print("掩码图像分析")
print("=" * 60)
print(f"掩码图像 shape: {mask.shape}")
print(f"掩码图像 dtype: {mask.dtype}")
print(f"掩码图像值范围：[{mask.min()}, {mask.max()}]")

# 方法 1：使用 torch.unique() 获取所有唯一值
unique_values = torch.unique(mask)
print(f"\n掩码中的所有唯一像素值：{unique_values.tolist()}")
print(f"唯一值的数量：{len(unique_values)}")

# 方法 2：更详细的统计信息
print("\n" + "=" * 60)
print("详细统计信息")
print("=" * 60)

# 将 mask 转换为一维数组便于统计
mask_flat = mask.flatten()

# 统计每个值出现的次数
for value in unique_values:
    count = (mask == value).sum().item()
    percentage = (count / mask.numel()) * 100

    if value == 0:
        print(f"像素值 {value:3d}: 出现 {count:6d} 次，占比 {percentage:6.2f}% [背景]")
    else:
        print(f"像素值 {value:3d}: 出现 {count:6d} 次，占比 {percentage:6.2f}% [行人 ID={value}]")

# 方法 3：使用集合 (set) 来存储唯一值
unique_set = set(mask.flatten().tolist())
print(f"\n使用 Python set 存储的唯一值：{sorted(unique_set)}")

# 可视化展示
plt.figure(figsize=(20, 5))

# 子图 1：原始图像
plt.subplot(1, 5, 1)
plt.title("Original Image")
plt.imshow(image.permute(1, 2, 0))
plt.axis('off')

# 子图 2：原始掩码（看起来很黑）
plt.subplot(1, 5, 2)
plt.title("Raw Mask\n(looks black)")
plt.imshow(mask.permute(1, 2, 0))
plt.axis('off')

# 子图 3：增强对比度的掩码
plt.subplot(1, 5, 3)
plt.title("Enhanced Mask\n(values × 50)")
enhanced_mask = mask.float() * 50
plt.imshow(enhanced_mask.permute(1, 2, 0).clamp(0, 255).byte())
plt.axis('off')

# 子图 4：使用伪彩色显示
plt.subplot(1, 5, 4)
plt.title("Mask with Colormap")
plt.imshow(mask.permute(1, 2, 0)[0], cmap='tab10')
plt.axis('off')

# 子图 5：直方图显示像素分布
plt.subplot(1, 5, 5)
plt.title("Pixel Value Distribution")
if len(unique_values) <= 20:
    # 如果唯一值较少，用条形图
    plt.bar(range(len(unique_values)),
            [(mask == v).sum().item() for v in unique_values],
            tick_label=[str(v.item()) for v in unique_values])
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
else:
    # 如果唯一值较多，用直方图
    plt.hist(mask_flat.numpy(), bins=50)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# 额外信息：计算行人数量
num_pedestrians = len(unique_values) - 1  # 减去背景 (0)
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print(f"这张图片中有 {num_pedestrians} 个行人实例")
if num_pedestrians > 0:
    print(f"行人 ID 分别为：{unique_values[unique_values > 0].tolist()}")

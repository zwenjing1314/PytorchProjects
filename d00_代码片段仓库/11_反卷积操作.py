import torch
import torch.nn as nn

# 定义转置卷积层：输入16通道，输出33通道，3x3核，stride=2（上采样2倍）
conv_transpose = nn.ConvTranspose2d(16, 33, 3, stride=2)

# 输入：batch_size=20, 16通道, 50×100
input = torch.randn(20, 16, 50, 100)
output = conv_transpose(input)
print(output.shape)  # torch.Size([20, 33, 93, 100])

# 用于恢复原始尺寸（如 U-Net 中）
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

x = torch.randn(1, 16, 12, 12)
h = downsample(x)  # 尺寸减半: [1, 16, 6, 6]
recovered = upsample(h, output_size=x.size())  # 强制恢复为 [1, 16, 12, 12]
print(recovered.size())  # torch.Size([1, 16, 12, 12])

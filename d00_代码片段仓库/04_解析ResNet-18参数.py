import torchvision.models as models
import torch.nn as nn


def print_model_structure(model):
    """打印模型的详细结构"""

    print("\n" + "=" * 80)
    print("ResNet-18 完整结构")
    print("=" * 80)

    # 1. 初始卷积层
    print("\n【1】初始卷积层和归一化层")
    print(f"  conv1: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)")
    print(f"    - weight shape: {model.conv1.weight.shape}")
    print(f"    - 参数量：{model.conv1.weight.numel():,}")

    print(f"  bn1: nn.BatchNorm2d(64)")
    print(f"    - weight shape: {model.bn1.weight.shape}")
    print(f"    - bias shape: {model.bn1.bias.shape}")
    print(f"    - running_mean: {model.bn1.running_mean.shape}")
    print(f"    - running_var: {model.bn1.running_var.shape}")
    print(f"    - 可学习参数量：{model.bn1.weight.numel() + model.bn1.bias.numel():,}")

    print(f"  relu: nn.ReLU(inplace=True)")
    print(f"  maxpool: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)")

    # 2. 4 个残差块组
    print("\n【2】残差块组（4 个 stage）")

    for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        stage = getattr(model, stage_name)
        print(f"\n  {stage_name}:")

        for block_idx, block in enumerate(stage):
            print(f"    [{block_idx}] BasicBlock:")

            # 每个 BasicBlock 包含 2 个卷积层
            print(f"      conv1: nn.Conv2d({block.conv1.in_channels}, {block.conv1.out_channels}, "
                  f"kernel_size=3, stride={block.conv1.stride[0]}, padding=1, bias=False)")
            print(f"        - weight shape: {block.conv1.weight.shape}")

            print(f"      bn1: nn.BatchNorm2d({block.bn1.num_features})")
            print(f"        - weight shape: {block.bn1.weight.shape}")
            print(f"        - bias shape: {block.bn1.bias.shape}")

            print(f"      conv2: nn.Conv2d({block.conv2.in_channels}, {block.conv2.out_channels}, "
                  f"kernel_size=3, stride=1, padding=1, bias=False)")
            print(f"        - weight shape: {block.conv2.weight.shape}")

            print(f"      bn2: nn.BatchNorm2d({block.bn2.num_features})")
            print(f"        - weight shape: {block.bn2.weight.shape}")
            print(f"        - bias shape: {block.bn2.bias.shape}")

            # 如果有下采样（shortcut 连接）
            if block.downsample is not None:
                print(f"      downsample:")
                print(f"        - 0: nn.Conv2d({block.downsample[0].in_channels}, "
                      f"{block.downsample[0].out_channels}, kernel_size=1, bias=False)")
                print(f"          weight shape: {block.downsample[0].weight.shape}")
                print(f"        - 1: nn.BatchNorm2d({block.downsample[1].num_features})")
                print(f"          weight shape: {block.downsample[1].weight.shape}")
                print(f"          bias shape: {block.downsample[1].bias.shape}")

    # 3. 全局平均池化和全连接层
    print("\n【3】全局平均池化和分类层")
    print(f"  avgpool: nn.AdaptiveAvgPool2d((1, 1))")
    print(f"  fc: nn.Linear({model.fc.in_features}, {model.fc.out_features})")
    print(f"    - weight shape: {model.fc.weight.shape}")
    print(f"    - bias shape: {model.fc.bias.shape}")


if __name__ == '__main__':
    # 创建和你一样的模型
    model = models.resnet18(pretrained=False)  # 为了快速演示，不用预训练
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)

    print("=" * 80)
    print("方法 1: 使用 model.parameters() - 查看所有可学习参数")
    print("=" * 80)

    total_params = 0
    for i, param in enumerate(model.parameters()):
        print(f"\n参数 {i}:")
        print(f"  shape: {param.shape}")
        print(f"  元素数量：{param.numel()}")
        print(f"  requires_grad: {param.requires_grad}")
        total_params += param.numel()

    print(f"\n{'=' * 80}")
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"{'=' * 80}")

    model1 = models.resnet18(pretrained=False)  # 加载预训练模型
    print_model_structure(model1)
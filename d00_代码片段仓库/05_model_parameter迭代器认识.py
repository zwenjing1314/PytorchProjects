# # 一、model.parameters() 是迭代器
#
# import torch.nn as nn
#
# # 简单模型
# model = nn.Sequential(
#     nn.Linear(10, 5),
#     nn.ReLU(),
#     nn.Linear(5, 2)
# )
#
# # 1. 查看类型
# params_iter = model.parameters()
# print(f"type: {type(params_iter)}")
# # <class 'list_iterator'> ← 确实是一个迭代器
#
# # 2. 可以遍历
# print("\n=== 遍历参数 ===")
# for i, param in enumerate(model.parameters()):
#     print(f"参数 {i}: shape={param.shape}")
#
# # 3. 可以转换为列表
# params_list = list(model.parameters())
# print(f"\n参数数量：{len(params_list)}")
#
# # 4. 只能遍历一次！
# params_iter2 = model.parameters()
# first_pass = list(params_iter2)
# second_pass = list(params_iter2)
# print(f"\n第一次遍历：{len(first_pass)} 个参数")
# print(f"第二次遍历：{len(second_pass)} 个参数")  # 0！迭代器已耗尽

#
# # 二、参数保存在各个层的属性中！
#
# import torch.nn as nn
#
# # 创建一个简单的模型
# linear = nn.Linear(3, 2, bias=True)
#
# print("=== linear 对象的属性 ===")
# print(dir(linear))
# # ['__class__', ..., 'bias', 'weight', ...]
#
# print("\n=== weight 属性 ===")
# print(f"type(weight): {type(linear.weight)}")
# # <class 'torch.nn.parameter.Parameter'>
#
# print(f"weight.shape: {linear.weight.shape}")
# # torch.Size([2, 3])
#
# print(f"weight.data:\n{linear.weight.data}")
# # tensor([[...], [...]])
#
# print("\n=== bias 属性 ===")
# print(f"type(bias): {type(linear.bias)}")
# # <class 'torch.nn.parameter.Parameter'>
#
# print(f"bias.shape: {linear.bias.shape}")
# # torch.Size([2])
#
# print(f"bias.data:\n{linear.bias.data}")
# # tensor([...])


# # 三、关键点：nn.Parameter 类
# from torch.nn import Parameter
# import torch
#
# # nn.Parameter 是 torch.Tensor 的子类
# print(f"Parameter 的父类：{Parameter.__bases__}")
# # (<class 'torch.Tensor'>,)
#
# # 创建 Parameter
# weight_data = torch.randn(2, 3)
# weight_param = Parameter(weight_data)
#
# print(f"\nParameter 特性:")
# print(f"requires_grad: {weight_param.requires_grad}")  # True ← 自动设置为 True
# print(f"is_leaf: {weight_param.is_leaf}")  # True ← 是叶子节点
#
# # 普通张量 vs Parameter
# ordinary_tensor = torch.randn(2, 3, requires_grad=True)
# param_tensor = Parameter(torch.randn(2, 3))
#
# print(f"\n普通张量的 requires_grad: {ordinary_tensor.requires_grad}")  # True
# print(f"Parameter 的 requires_grad: {param_tensor.requires_grad}")  # True
#
# # 关键区别：Parameter 会被 module.parameters() 自动收集！


# 四、架构保存在模块的 _modules 字典中
import torch
import torch.nn as nn


# 定义一个自定义网络
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


model = MyNet()

print("=== 模型的内部结构 ===")
print(f"\n1. _modules (子模块字典):")
print(model._modules)
# OrderedDict([
#     ('conv1', Conv2d(...)),
#     ('bn1', BatchNorm2d(...)),
#     ('relu', ReLU()),
#     ('fc1', Linear(...))
# ])

print(f"\n2. _parameters (直接参数):")
print(model._parameters)
# OrderedDict() ← 空，因为参数都在子模块里

print(f"\n3. _buffers (缓冲寄存器):")
print(model._buffers)
# OrderedDict() ← 空，缓冲在子模块里

# 访问子模块的参数
print(f"\n4. conv1 的参数:")
print(f"   conv1._parameters: {model.conv1._parameters.keys()}")
# odict_keys(['weight', 'bias'])

print(f"   conv1.weight.shape: {model.conv1.weight.shape}")
# torch.Size([64, 3, 3, 3])

print("*" * 80)


# 完整的属性层次结构
def show_module_hierarchy(module, indent=0):
    """递归显示模块的层次结构"""

    prefix = "  " * indent

    # 打印当前模块
    print(f"{prefix}{type(module).__name__} (id: {id(module)})")

    # 打印直接参数
    for name, param in module._parameters.items():  # 遍历字典
        if param is not None:
            print(f"{prefix}  ├─ [Parameter] {name}: {tuple(param.shape)}")

    # 打印缓冲
    for name, buf in module._buffers.items():
        if buf is not None:
            print(f"{prefix}  ├─ [Buffer] {name}: {tuple(buf.shape)}")

    # 递归打印子模块
    for name, child in module._modules.items():
        print(f"{prefix}  └─ [Module] {name}:")
        show_module_hierarchy(child, indent + 2)


from torchvision import models

# 使用
show_module_hierarchy(model)

if __name__ == '__main__':
    model = models.resnet18(pretrained=False)
    print(model)
    print(type(model))  # <class 'torchvision.models.resnet.ResNet'>

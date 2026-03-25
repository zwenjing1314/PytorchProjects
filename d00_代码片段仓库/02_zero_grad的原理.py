# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # 1. 创建模型
# model = nn.Linear(3, 2, bias=True)
# # 结构：y = x * W^T + b
# # W shape: (2, 3), b shape: (2,)
#
# # 2. 查看参数（权重）
# print("=== 模型参数 ===")
# print(f"weight:\n{model.weight}")
# # tensor([[ 0.1, -0.2,  0.3],
# #         [-0.1,  0.4, -0.5]], requires_grad=True)
# print(f"bias:\n{model.bias}")
# # tensor([0.2, -0.3], requires_grad=True)
#
# # 3. 初始时梯度为 None
# print("\n=== 初始梯度 ===")
# print(f"weight.grad: {model.weight.grad}")  # None
# print(f"bias.grad: {model.bias.grad}")      # None
#
# # 4. 前向传播 + 反向传播
# x = torch.tensor([[1.0, 2.0, 3.0]])  # 输入
# target = torch.tensor([[1.0, 0.0]])  # 目标值
#
# output = model(x)                    # 前向传播
# loss = nn.MSELoss()(output, target)  # 计算损失
# loss.backward()                      # 反向传播 ← 梯度计算并存储
#
# # 5. 现在梯度有值了！
# print("\n=== 反向传播后的梯度 ===")
# print(f"weight.grad:\n{model.weight.grad}")
# print(f"weight:\n{model.weight}")
#
# print(f"bias.grad:\n{model.bias.grad}")
# print(f"bias:\n{model.bias}")
#
# # 6. 梯度存储在参数的 .grad 属性中
# print("\n=== 梯度的存储位置 ===")
# print(f"梯度存储在 model.weight.grad: {id(model.weight.grad)}")
# print(f"梯度存储在 model.bias.grad: {id(model.bias.grad)}")


import torch
import torch.nn as nn
import torch.optim as optim

# 创建模型
model = nn.Linear(3, 2, bias=False)  # 为了简单，不用偏置

# 创建优化器（优化器会记录要优化的参数）
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("=== 步骤 1: 初始状态 ===")
print(f"weight:\n{model.weight.data}")
print(f"weight.grad: {model.weight.grad}")  # None

# === 第一次迭代 ===
print("\n=== 步骤 2: 第一次前向 + 反向传播 ===")
x1 = torch.tensor([[1.0, 0.0, 0.0]])
target1 = torch.tensor([[1.0, 0.0]])

output1 = model(x1)
loss1 = nn.MSELoss()(output1, target1)
loss1.backward()  # 计算梯度

print(f"第一次 loss: {loss1.item():.4f}")
print(f"weight.grad:\n{model.weight.grad}")
# tensor([[ 0.,  0.,  0.],
#         [-0., -0., -0.]])

# === 第二次迭代（不清零）===
print("\n=== 步骤 3: 第二次前向 + 反向传播（不 zero_grad）===")
x2 = torch.tensor([[0.0, 1.0, 0.0]])
target2 = torch.tensor([[0.0, 1.0]])

output2 = model(x2)
loss2 = nn.MSELoss()(output2, target2)
loss2.backward()  # 再次计算梯度 ← 梯度会累加！

print(f"第二次 loss: {loss2.item():.4f}")
print(f"weight.grad:\n{model.weight.grad}")
# ⚠️ 注意：梯度是两次反向传播的累加！
# tensor([[ 0.,  0.,  0.],
#         [ 0., -2.,  0.]])  ← 这里累加了

# === 使用 zero_grad() 清零 ===
print("\n=== 步骤 4: 使用 optimizer.zero_grad() ===")
optimizer.zero_grad()  # 将所有参数的 .grad 设为 0

print(f"zero_grad 后 weight.grad:\n{model.weight.grad}")
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# === 第三次迭代（先清零）===
print("\n=== 步骤 5: 第三次前向 + 反向传播（先 zero_grad）===")
x3 = torch.tensor([[0.0, 0.0, 1.0]])
target3 = torch.tensor([[1.0, 1.0]])

optimizer.zero_grad()  # 先清零 ← 关键步骤！
output3 = model(x3)
loss3 = nn.MSELoss()(output3, target3)
loss3.backward()

print(f"第三次 loss: {loss3.item():.4f}")
print(f"weight.grad:\n{model.weight.grad}")
# ✅ 只有第三次的梯度，没有累加
# tensor([[ 0.,  0.,  0.],
#         [ 0.,  0., -2.]])

# 创建模型
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 优化器内部记录了所有要优化的参数
print("=== 优化器管理的参数 ===")
for i, param_group in enumerate(optimizer.param_groups):
    print(f"参数组 {i}:")
    print(f"  学习率：{param_group['lr']}")
    print(f"  参数数量：{len(param_group['params'])}")

    for j, param in enumerate(param_group['params']):
        print(f"    参数 {j}: shape={param.shape}, requires_grad={param.requires_grad}")

# 输出：
# 参数组 0:
#   学习率：0.01
#   参数数量：4
#     参数 0: shape=torch.Size([5, 10])  ← Linear(10,5)的weight
#     参数 1: shape=torch.Size([5])      ← Linear(10,5)的bias
#     参数 2: shape=torch.Size([2, 5])   ← Linear(5,2)的weight
#     参数 3: shape=torch.Size([2])      ← Linear(5,2)的bias

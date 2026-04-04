import torch

# 创建两个形状相同的张量
a = torch.tensor([[1, 2, 3],])
b = torch.tensor([[4, 5, 6],])

# 在新维度 dim=0 堆叠
stack0 = torch.stack([a, b], dim=0)
print(stack0)
print(stack0.shape)  # torch.Size([2, 3])

# 在新维度 dim=1 堆叠
stack1 = torch.stack([a, b], dim=1)
print(stack1)
print(stack1.shape)  # torch.Size([3, 2])

stack1 = torch.stack([a, b], dim=2)
print(stack1)
print(stack1.shape)  # torch.Size([3, 2])

"""
tensor([[1, 2, 3],
        [4, 5, 6]])
torch.Size([2, 3])

tensor([[1, 4],
        [2, 5],
        [3, 6]])
torch.Size([3, 2])

"""
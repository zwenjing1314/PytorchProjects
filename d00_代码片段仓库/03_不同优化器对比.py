import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import torch
import torch.optim as optim


# 创建一个简单的损失函数 landscape
def loss_function(x, y):
    return (x ** 2 + y ** 2)  # 简单的碗状曲面


# 比较三种优化器
optimizers_config = [
    ('SGD', optim.SGD, {'lr': 0.1}),
    ('SGD+Momentum', optim.SGD, {'lr': 0.1, 'momentum': 0.9}),
    ('Adam', optim.Adam, {'lr': 0.1}),
]

results = {}

for name, OptClass, kwargs in optimizers_config:
    # 重置参数
    x = torch.tensor([5.0], requires_grad=True)
    y = torch.tensor([5.0], requires_grad=True)
    optimizer = OptClass([x, y], **kwargs)

    trajectory_x = [x.item()]
    trajectory_y = [y.item()]

    # 优化 20 步
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_function(x, y)
        loss.backward()
        optimizer.step()

        trajectory_x.append(x.item())
        trajectory_y.append(y.item())

    results[name] = (trajectory_x, trajectory_y)

# 可视化对比
plt.figure(figsize=(10, 8))
x_grid = np.linspace(-6, 6, 100)
y_grid = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = X ** 2 + Y ** 2
plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)

for name, (traj_x, traj_y) in results.items():
    plt.plot(traj_x, traj_y, 'o-', label=name, linewidth=2, markersize=8)

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('不同优化器的收敛路径对比')
plt.colorbar(label='Loss')
plt.show()
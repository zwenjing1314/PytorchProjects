import torch
import torch.nn as nn
import torch.nn.functional as F


class MyFNN(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        构建一个简单的前馈神经网络

        参数:
            D_in: 输入层维度
            H: 隐藏层维度
            D_out: 输出层维度
        """
        super(MyFNN, self).__init__()
        # 第一层：输入层 -> 隐藏层
        self.linear1 = nn.Linear(D_in, H)
        # 第二层：隐藏层 -> 输出层
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        """
        前向传播过程

        数据流向：
        1. 输入数据 x 进入第一层线性层
        2. 经过 ReLU 激活函数引入非线性
        3. 进入第二层线性层得到最终输出
        """
        # 通过第一层并应用 ReLU 激活函数
        x = self.linear1(x)
        x = F.relu(x)
        # 通过第二层得到输出
        x = self.linear2(x)
        return x


# 设置网络参数
D_in = 10  # 输入特在维度
H = 20  # 隐藏层神经元数量
D_out = 2

# 创建模型实例
model = MyFNN(D_in, H, D_out)

# 创建一些测试数据
x = torch.randn(1, D_in)  # 批次大小为 1 的随即输入

# 设置模型为评估模式
model.eval()

# 打印保存新前模型的预测值
with torch.no_grad():
    print("保存当前的预测值：")
    print(model(x).reshape(-1)[:5])

# 保存模型状态字典到磁盘
torch.save(model.state_dict(), "model_state_dict.pth")
print("\n模型已保存到 model_state_dict.pth")

# 加载模型：从磁盘加载状态字典
state_dict = torch.load("model_state_dict.pth")

# 手动创建新的网络实例（需要与原始网络结构相同）
saved_model = MyFNN(D_in, D_out, H)  # 注意：这里故意传错参数顺序来演示问题
saved_model.eval()  # 将 Dropout、Batch Normalization 层设置为评估模式

# 将状态字典应用到网络模型，并打印加载后模型的预测值
# 这行会报错，因为参数形状不匹配，用于演示 state_dict 的重要性
try:
    saved_model.load_state_dict(state_dict)
    with torch.no_grad():
        print("\n加载后的预测值:")
        print(saved_model(x).reshape(-1)[:5])
except RuntimeError as e:
    print(f"\n错误演示：{e}")
    print("原因：网络结构参数不匹配")

# 正确的做法：创建相同结构的网络
print("\n--- 正确的加载方式 ---")
correct_model = MyFNN(D_in, H, D_out)  # 正确的参数顺序
correct_model.load_state_dict(torch.load('model_state_dict.pth'))
correct_model.eval()

with torch.no_grad():
    print("正确加载后的预测值:")
    print(correct_model(x).reshape(-1)[:5])

# 验证两个模型的预测结果是否一致
print("\n--- 验证一致性 ---")
model.eval()
correct_model.eval()
with torch.no_grad():
    pred1 = model(x).reshape(-1)[:5]
    pred2 = correct_model(x).reshape(-1)[:5]
    print(f"原始模型预测：{pred1}")
    print(f"加载模型预测：{pred2}")
    print(f"预测结果是否一致：{torch.allclose(pred1, pred2)}")

print(model)  # <class '__main__.MyFNN'>
print(type(model))
print(model.state_dict())
print(type(model.state_dict()))  # <class 'collections.OrderedDict'>

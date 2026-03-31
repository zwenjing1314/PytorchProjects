"""
广播是怎么发生的？
先假设一个非常小的例子，方便你在脑子里“算”一遍。

1. 原始 mask（一张 H×W 的图）
假设 mask 形状是 (3, 3)，内容是（为简单起见只看单通道）：

mask =
[[0, 1, 1],
 [0, 2, 2],
 [0, 2, 1]]
这里：

0 表示背景
1、2 表示两个不同行人实例
2. 得到 obj_ids
obj_ids = torch.unique(mask)
# tensor([0, 1, 2])
obj_ids = obj_ids[1:]
# tensor([1, 2])
# 形状是 (2,)
3. 变成 (N, 1, 1) 方便广播
obj_ids[:, None, None]
# 形状从 (2,) -> (2, 1, 1)
# 具体数值是：
[
 [[1]],
 [[2]],
]
4. 与 mask 比较时的广播规则
表达式：

masks = (mask == obj_ids[:, None, None])
此时：

mask 的形状是 (3, 3)，等价于 (1, 3, 3)
obj_ids[:, None, None] 的形状是 (2, 1, 1)
按 PyTorch 广播规则逐维对齐：

第 1 维：1 和 2 → 扩展成 2
第 2 维：3 和 1 → 扩展成 3
第 3 维：3 和 1 → 扩展成 3
所以结果形状是 (2, 3, 3)，可以理解为：

第 0 维是“实例索引”（第几个 obj_id）
后面两维是空间位置（H, W）
5. 展开来看每一层
第 0 层：对应 obj_ids[0] = 1：

masks[0] = (mask == 1)
# =
[
 [False,  True,  True],
 [False, False, False],
 [False, False,  True],
]
第 1 层：对应 obj_ids[1] = 2：

masks[1] = (mask == 2)
# =
[
 [False, False, False],
 [False,  True,  True],
 [False,  True, False],
]
转成 uint8 后就是 0/1 的二值掩码：


masks.to(torch.uint8) =
[
 [[0, 1, 1],
  [0, 0, 0],
  [0, 0, 1]],   # 实例1的mask
 [[0, 0, 0],
  [0, 1, 1],
  [0, 1, 0]],   # 实例2的mask
]
"""

import torch

mask = torch.tensor(
    [[0, 1, 1],
    [0, 2, 2],
    [0, 2, 1]]
)
obj_ids = torch.tensor([1, 2])

masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
print(masks)

"""
tensor([[[0, 1, 1],
         [0, 0, 0],
         [0, 0, 1]],

        [[0, 0, 0],
         [0, 1, 1],
         [0, 1, 0]]], dtype=torch.uint8)
"""
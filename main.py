import numpy as np
from sympy import num_digits
import torch

num_objs = 6

iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
print(iscrowd)
print(iscrowd.shape)
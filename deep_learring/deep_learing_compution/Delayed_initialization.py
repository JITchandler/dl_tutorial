## 延后初始化
## 当自己不知道输入维度该设定为多少的时候，我们采用延后初始化，使得框架自动的推断输入的形状，并且初始化权重参数
## 在python中，常常采用 nn.Lazylinear
import numpy as np
import torch
from torch import nn
net = nn.Sequential(nn.LazyLinear(256),nn.ReLU(),nn.LazyLinear(10))
print("尚未初始化",net[0].weight)
print("未初始化的模型：",net)
x = torch.rand(2,25)
net(x)
print("延后初始化的模型：",net)
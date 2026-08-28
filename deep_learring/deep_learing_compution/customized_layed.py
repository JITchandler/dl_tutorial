import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
## 我们有时候需要我们构建自定义层
## 1.不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x - x.mean()
# layer = CenteredLayer()
# out = layer(torch.FloatTensor([1,2,3,4,5])) ## 注意，这里要写出张量的形式
# print(out)

## 现在我们将上面的层，作为组件合并到更为复杂的模型中
net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y = net(torch.rand(4,8))
# print(Y.mean())

## 定义带有参数的层
class MyLinear(nn.Module):
    def __init__(self,in_units,units): ## 输入特征维度，输出特征维度
        super().__init__()
        ## 定义可学习的参数
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units))
    def forward(self,x):
        linear = torch.matmul(x,self.weight) + self.bias
        return F.relu(linear)
## 我们可以使用自定义层直接执行前向传播计算
linear = MyLinear(5,3)
print(linear.weight)
# out = linear(torch.rand(2,5))
# # print(out)
## 我们还可以使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层
net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
out = net(torch.rand(2,64))
print(out)



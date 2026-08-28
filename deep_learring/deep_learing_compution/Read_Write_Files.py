import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
## 这一章主要是学习如何保存训练的模型和以及在模型在负责训练过程中的结果，便于下次使用
## 1.加载和保存张量(tensor)
x = torch.arange(4)
torch.save(x,"x-file")
x2 = torch.load("x-file") ## 可以将存储在文件中的数据读回内存
##print(x2)
## 我们可以存储一个张量列表，然后将他们读回内存
y = torch.zeros(4)
torch.save([x,y],"x-files")
x2 ,y2 = torch.load("x-files")
##print(x2,y2)
## 我们也可以写入或者读取从字符串中映射到张量的字典，当我们要读取和或者写入模型中的所有权重的时候，这很方便
mydict = {"x":x,"y":y}
torch.save(mydict,"mydict")
mydict2 = torch.load("mydict")
##print(mydict2)
## 加载和保存模型参数
## 光是保存权重向量还不够，当模型复杂的时候，权重向量的数量将会非常的惊人
## 为此，深度学习框架提供了内置函数来保存和加载整个网络
## 注意，保存的是模型的参数而不是整个模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)
    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
x = torch.randn(size =(2,20))
y = net(x)
print(y)
## 接下来我们将模型的参数存储在一个叫做“mlp.params”文件中
torch.save(net.state_dict(),"mlp.params")
## 为了恢复模型，我们实例化了原始多层感知机模型的一个备份，这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
print(clone.eval())
Y_clone = clone(x)
print(Y_clone == y)




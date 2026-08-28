import torch
from nltk.corpus.reader import lin
from sympy.geometry import line
from  torch import nn
from torch.nn import functional as F
## 通过实例化nn.Sequential来构建我们的模型
net = nn.Sequential(
    nn.Linear(20,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
x = torch.randn(2,20)

## 自定义块：
## 首先区别一下在神经网络中，层与块的区别，层是最小的基础单元，做单一的固定操作，如线性变换，卷积，激活，池化等
## 而块是多个层或者是子块打包而成的组合
## 其中块的基本功能有如下：
## 1.将输入数据作为前向传播的参数
## 2.通过前向传播函数来生成输出，注意，输出的形状和输入的形状不一定一样
## 3，计算其输出关于输入的梯度，可以通过反向传播来进行访问，通常是自动发生的
## 4.存储和访问前向传播计算所需的参数
## 5.根据需要初始化模块
class MLP(nn.Module): # -> 块
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256) # -> 全连接层 隐藏层
        self.out = nn.Linear(256,10) # -> 全连接 输出层

    def forward(self,x):
        return self.out(F.relu(self.hidden(x)))

# net = MLP()
# print(net(x))

## 顺序块
## 定义自己Sequential，需要两个关键的函数
## 1.一种将块逐个追加到列表中的函数
## 2.一种前向传播的函数，用于将输入按追加块的顺序传递给块组成的“链条”

class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx,module in enumerate(args):
            self._modules[str(idx)] = module  # 将放进来的 层/块 按顺序存进_modules里面

    # ✅ 修复：forward 必须和 __init__ 对齐，同级！
    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net(x))

## 下面我们举一个例子，然我们来看看块有多灵活
## 块中可有不用参加训练的常量参数
## 块中可以复用同一个层
## 块中可以自定义任意的数学运算
## 块中可以用python 控制流

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self._rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self,x):
        x = self.linear(x)
        x = F.relu(torch.mm(x,self._rand_weight) + 1)
        x =self.linear(x)
        while x.abs().sum()>1:
            x /=2
        return x.sum()
net = FixedHiddenMLP()
# print(net(x))

## 层与块的嵌套使用
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self,x):
        return self.linear(self.net(x))
chimera = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
print(chimera(x))

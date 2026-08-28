## 这一章主要是讲解参数管理，
## 重要介绍以下内容
## 访问参数，用于调试，诊断和可视化
## 参数初始化
## 在不同模型组件之间共享参数
import numpy as np
import torch
from torch import nn
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
x= torch.rand(size=(2,4))
print(net(x))
print("输出层",net[2].state_dict()) ## 通过索引来访问模型的任意层，目前访问的是最后一层的输出层,显示权重和偏置的具体量
print("激活函数",net[1].state_dict()) ## 激活函数里面没有参数


## 目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None) ## 由于此模型没有调用反向函数，故参数的梯度处于初始状态

## 一次性访问所有参数
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
print(*[(name,param.shape) for name,param in net.named_parameters()])
## 从嵌套块中收集参数
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),
                         nn.Linear(8,4),nn.ReLU(), )
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())
    return net
rgnet = nn.Sequential(block2(),nn.Linear(4,1))
print("嵌套块：",rgnet(x))
## 参数初始化，深度学习框架默认是随机初始化，但是也允许我们自定义初始化方法
## 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print("内置的初始化器：",net[0].weight.data[0],net[0].bias.data[0])

## 我们也可以将参数初始化为常数
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1) ## 权重设置为1
        nn.init.zeros_(m.bias) ## 偏置设置为0
net.apply(init_constant)
print("内置的初始化为常数：",net[0].weight.data[0],net[0].bias.data[0])

## 我们还可以对某些块应用不同的初始化方法
def init_Xavier(m): ## 使用Xavier初始化方法初始化第一个神经网络层
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m): ## 第三个神经网络层初始化为常量值42
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(init_Xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

## 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        ## 打印正在初始化的参数
        print("Init",*[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])

## 当然也可以始终直接设置参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

## 参数绑定
## 参数绑定指的是多个层使用同一个nn.module对象，参数是同一份地址
## 修改其中的一层的参数，另一个层的参数也会同时改变
## 在反向传播的时候，来自两个层的梯度会自动累加，一起更新这一份共享参数
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    nn.Linear(8,1),
                    )
net(x)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0,0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])


import matplotlib.pyplot as plt
import torch
import  random
import numpy as np
from torch import autograd


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    x = np.random.normal(0, 1, (num_examples, len(w)))
    ## 生产均值0，标准差为1的正态分布随机数
    y = np.dot(x, w) + b
    ## 矩阵乘法
    y += np.random.normal(0, 0.01, y.shape)
    return x , y.reshape((-1, 1))

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
## 随机读取小批量的数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
## 定义初始化模型参数
w = torch.normal(0, 0.01, (2, 1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

## 定义模型
def linreg(x, w, b):
     return torch.matmul(x, w) + b

## 定义损失函数 :平方误差
def squared_loss(y_hat, y):
    return (y_hat -y.reshape(y_hat.shape))**2 / 2

## 优化算法，小批量随机梯度下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
         param -= lr * param.grad / batch_size
         param.grad.zero_()

lr = 0.03 # 学习率
num_epochs = 3 #训练轮数，把全部1000个数据看几遍
net = linreg #模型，方便后期的模型替换
loss = squared_loss  # 损失函数：均方误差

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            l = loss(net(x, w, b), y)  #记录计算过程，准备求梯度，X和y的小批量损失
            l.sum().backward()  # 反向传播求梯度 #算出：怎么改 w 和 b，能让损失变小,梯度存在 w.grad 和 b.grad 里
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(
            net(torch.tensor(features, dtype=torch.float32), w, b),
            torch.tensor(labels, dtype=torch.float32)
        )
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')




##Stochastic gradient descent
## 随机梯度下降

## 之前的标准梯度下降是 用 全部的样本算整体损失梯度，再更新参数，缺点是计算速记极慢

## 随机梯度下降：是每次只随机抽取 1 个样本，只用这单个样本的损失梯度更新参数
## 这样做可以使得计算速度快，噪声大，更新路线抖，泛化效果好，更容易跳出局部极小
import math

import d2l
import math
import torch
from d2l import torch as d2l
def train_2d(train,steps = 20,f_grad = None):
    ## s1,s2是内部状态
    x1, x2 ,s1, s2 = -5,-2,0,0
    results = [(x1,x2)]
    for i in range(steps):
        if f_grad:
            x1 , x2 , s1 ,s2 = train(x1,x2,s1,s2,f_grad) ## train 用来更新这四个参数
        else :
            x1, x2, s1, s2 = train(x1, x2, s1, s2)
        results.append((x1,x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
def show_trace_2d(f, results):  #@save
    """显示优化过程中2D变量的轨迹"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                          torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    d2l.plt.show()


def f(x1,x2):
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1,x2):
    return 2 * x1, 4 * x2

def constant_lr():
    return 1
eta = 0.1
lr = constant_lr
def sgd(x1,x2,s1,s2,f_grad):
    g1,g2 = f_grad(x1,x2)
    # 给梯度叠加高斯模糊，N(0,1)，模拟单样本随机梯度
    g1 += torch.normal(0.0,1,(1,)).item()
    g2 += torch.normal(0.0,1,(1,)).item()
    eta_t = eta * lr()
    return (x1 -  eta_t * g1 , x2 - eta_t * g2,0,0)
# show_trace_2d(f,train_2d(sgd,steps = 50, f_grad = f_grad))
## 随机梯度下降全程充满了随机性质，即使接近了最小值，还是具有不确定性，更加糟糕的是，就算得到了额外的帮助，也不会有太多的改善
## 唯一解决的方法就是改变学习率，更准确的是动态降低学习率


## 动态调整学习率
## 动态学习率的核心思想：训练前期学习率大，训练后期学习率小，随迭代步数自动变化
## 常见的动态学习率有：常数衰减，分段衰减，指数衰减，多项式衰减
t = 1
def expontial_lr():
    global t
    t += 1
    return math.exp(-0.1* t)

lr = expontial_lr
# show_trace_2d(f,train_2d(sgd,steps = 1000, f_grad = f_grad))
## 这种方法的方差大大减少，但还是没能收敛到最优解，该算法无法收敛
## 另一方面，我们使用多项式衰减，其中学习率随着迭代次数的平方根倒数衰减，在50次迭代之后，收敛就会更好

def polynomial_lr():
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)
lr = polynomial_lr
show_trace_2d(f,train_2d(sgd,steps = 1000, f_grad = f_grad))

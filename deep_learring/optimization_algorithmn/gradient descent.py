## 第三节 梯度下降  gradient descent

import numpy as np
import torch
from d2l import torch as d2l

def f(x):  # 目标函数
    return x ** 2

def f_grad(x):  # 目标函数的梯度(导数)
    return 2 * x

def gd (eta,f_grd):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grd(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results
results = gd(0.2,f_grad)

## 对x进行优化的过程
def show_trace(results,f):
    n = max(abs(min(results)),abs(max(results)))
    f_line = torch.arange(-n,n,0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])
    d2l.plt.show()

# show_trace(results, f)
## 学习率
## 学习率是用力来决定目标函数是否能够收敛到局部最小值，以及何时收敛到局部最小值
## 学习率过小，会导致目标函数收敛的速度过慢，学习率过大，会跳过局部最小值

# show_trace(gd(0.05,f_grad), f)
## 学习率过大，则会导致发散
# show_trace(gd(1.1, f_grad), f)

## 局部最小值
## 当一个函数有多个最小值的时候，根据我们的学习率，我们最终只能得到许多解的一个，
## 下面的例子说明了高学习率如何导致较差的局部最小值
c = torch.tensor(0.15 * np.pi)

def f(x): ## 目标函数的梯度
    return x * torch.cos(c * x)
def f_grad(x): ## 目标函数的梯度
    return torch.cos(c * x) - c * x * torch.sin(c * x)

# show_trace(gd(2,f_grad), f)

## 多元梯度下降

## 我们来看看多元函数梯度下降的情况，
## 构造目标函数：f = x1^2 + 2x2^2

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
## 我们来观察学习率为 0.1 时候优化变量x的轨迹，时间步结束之后，x的接近其位于[0,0]的最小值
## 虽然进展顺利，但是缓慢

def f_2d(x1,x2): ## 目标函数
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1,x2): ## 目标函数的梯度
    return (2 * x1, 4 * x2)

def gd_2d(x1,x2,s1,s2,f_grad, eta = 0.1):
    g1,g2 = f_grad(x1,x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

show_trace_2d(f_2d, train_2d(gd_2d,steps= 20,f_grad=f_2d_grad))

## 自适应方法
## 从上面可以看出，学习率过大，过小都不行，能否自动的确定学习率呢
## 除了考虑目标函数的值和梯度，还需要考虑它的曲率的二阶方法可以帮助我们解决这个问题，

##  牛顿法：
##  牛顿法采用二阶海森矩阵替代固定学习率，根据损失曲率自动调整每一步的更新幅度，是理论完美的自适应二阶优化
## 但是计算成本较高，很少用于深度学习

c = torch.tensor(0.5)

## 这里设置一个凸双曲余弦函数c
def f(x): # 目标函数
    return torch.cosh(c * x )

def f_grad(x):## 目标函数的梯度
    return c * torch.sinh(c * x)
def f_hess(x):# 目标函数的hessian 也就是二阶导
    return c**2 * torch.cosh(c * x)
def newton(eta = 1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)


## 如果采用一个非凸函数，在牛顿法中，我们最终采用的是除以hessian 这就意味如果二阶导数是负的，则f的值有可能增加
## 这个是算法中的一个缺陷
c = torch.tensor(0.15 * np.pi)

def f(x): # 目标函数
    return x * torch.cos(c * x)
def f_grad(x):# 目标函数的hessian 也就是二阶导
    return torch.cos(c * x) - c * x * torch.sin(c * x)
def f_hess(x):  # 目标函数的Hessian
    return - 2 * c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)
show_trace(newton(), f)
## 采用较小的学习率
show_trace(newton(0.5), f)
## 第二小节 凸性 convexity

## 凸集和凸函数
## 凸集是凸性的基础，简单理解就是 在凸集中的两点相连的线段也在凸集之中
## 凸集的交集也是凸集;凸集的和，数乘，仿射变换后还是凸集，凸集任意图组合仍然是属于集合

## 凸函数 集合必须是凸集
## 几何定义：函数图像上两点连线永远在图像上方
import numpy as np
import torch
from d2l import torch as d2l
from mpl_toolkits import mplot3d
f = lambda x: 0.5 * x**2  # 凸函数
g = lambda x: torch.cos(np.pi * x)  # 非凸函数
h = lambda x: torch.exp(0.5 * x)  # 凸函数

x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
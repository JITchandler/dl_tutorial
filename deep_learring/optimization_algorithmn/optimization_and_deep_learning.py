## 第一节 ：优化和深度学习
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt

## 经验风险 g 和真实风险 f
## 真实风险：全体数据分布下的期望损失，是全局，平滑，无噪声的理想目标
## 经验风险：在有限数据集中计算得出的平均损失，有限样本自带采样噪声
## 因为有噪声项的存在，所以g不如f平滑
def f(x):
    return x * torch.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)

# 自定义注释函数（替换原d2l.plt.gca().annotate）
def annotate(text, xy, xytext):
    plt.gca().annotate(
        text,
        xy=xy,
        xytext=xytext,
        arrowprops=dict(arrowstyle='->')
    )

# 生成x张量
x = torch.arange(0.5, 1.5, 0.01)
y_f = f(x)
y_g = g(x)

# 替换d2l.set_figsize((4.5, 2.5))
plt.figure(figsize=(4.5, 2.5))

# 绘图，替换d2l.plot，同时转numpy适配matplotlib
plt.plot(x.numpy(), y_f.numpy(), label="risk f(x)")
plt.plot(x.numpy(), y_g.numpy(), label="empirical risk g(x)")

# 设置坐标轴标签（对应原d2l.plot的x,y参数）
plt.xlabel('x')
plt.ylabel('risk')

# 添加两处标注，和原代码坐标完全一致
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))



## 局部最小值
x = torch.arange(-1.0, 2.0, 0.01)
y = f(x)
plt.figure(figsize=(4.5, 2.5))
plt.plot(x.numpy(), y.numpy(), label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))

## 鞍点
## 鞍点是梯度消失的另一个原因，鞍点是指函数的所有指数的所有梯度都消失但既不是全局最小值也不是局部最小值的任意位置

x = torch.arange(-2.0,2.0,0.01)
y = f(x)
plt.figure(figsize=(4.5, 2.5))
plt.plot(x.numpy(), y.numpy(), label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
# 鞍点标注，坐标完全和原代码一致
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
plt.legend()
plt.tight_layout()
plt.show()

# 生成网格数据
x, y = torch.meshgrid(
    torch.linspace(-1.0, 1.0, 101),
    torch.linspace(-1.0, 1.0, 101)
)
z = x ** 2 - y ** 2

# 转换为numpy数组给matplotlib使用
x_np = x.numpy()
y_np = y.numpy()
z_np = z.numpy()

# 创建画布与3D坐标轴，替代d2l.plt.figure()
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')

# 绘制线框曲面
ax.plot_wireframe(x_np, y_np, z_np, rstride=10, cstride=10)
# 标记原点鞍点红色叉号
ax.plot([0], [0], [0], 'rx', markersize=10)

# 设置坐标轴刻度
ticks = [-1, 0, 1]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

# 设置坐标轴标签
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.tight_layout()
plt.show()

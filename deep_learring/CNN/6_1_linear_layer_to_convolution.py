## 第六章，从全连接层到卷积
## 如果采用全连接层来处理图像，会导致参数爆炸，而且效率极低
## 此外采用全连接层还会存在 权重不共享的问题，每个隐藏神经元都会有一套属于自己的权重

## 不变性：
## 平移不变性，在检测物体目标的时候，物体在图像中所在的位置，不影响检测的能力
## 局部性：为了收集相关的训练参数，把我们不应该偏离距离目标很远的地方


## 总结
## 卷积层将输入和卷积核进行交叉相关，加上偏移量得到输出
## 核矩阵和偏移是可学习的参数
## 核矩阵的大小是超参数
## 卷积神经网络（CNN）是一种特殊的神经网络，它可以包含多个卷积层
## 多个输入和输出通道使模型在每个空间位置可以获取图像的多个方面

## 图像卷积，代码实现

import torch
from torch import nn
## 手动实现二维互相关运算，就是课本中的卷积操作
def corr2d(X,K): ## 输入  卷积核
    h ,w = K.shape  ## 取出卷积核的高和宽
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) ## 算出输出的大小
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i ,j] = (X[i: i + h , j: j + w] * K).sum() ## 在输入中 切出一块和卷积核大小相同的区域，对于的数与卷积核进行相乘，之后求和
    return Y
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(corr2d(X,K))

## 卷积层
## 卷积层对于输入和卷积核权重进行互相关运算，并且添加偏置之后产生输出，所以其中的权重和偏置可以进行训练
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        return corr2d(x,self.weight) + self.bias
## 构造一个 6 * 8 像素的黑白图像，中间四列为黑色（0），其余像素为白色（1）
X  = torch.ones(6 ,8)
X [:,2:6] = 0
# print(X)
## 采用高度为1，宽度为2的卷积核 k，当进行互相关运算的时候，如果水平相邻的两元素相同，则输出为 0 ，否则输出为非 0
K = torch.tensor([[1.0,-1.0]])
Y = corr2d(X,K)
# print(Y)
## 从结果上面发现，输出中的 1 和 -1 分别表示白色到黑色的垂直边缘和黑色到白色的垂直边缘
# print(corr2d(X.t(),K))
## 结果发现原来的输入进行转置操作之后，原来的检测到的垂直边缘消失了，所以这个卷积核k只能检测到垂直边缘，无法检测到水平边缘

## 学习卷积核
## 接下来我们将学习如何自己动手训练一个卷积核
## 我们先构造一个卷积层，并且将卷积核初始化为随机张量，之后在每次迭代的时候，比较Y和卷积层输出的平方误差，然后计算梯度来更新卷积核
conv2d = nn.Conv2d(1,1,(1,2),bias=False)

X =X.reshape((1,1,6,8))
Y =Y.reshape((1,1,6,7))

lr = 3e-2

# for i in range(10):
#     Y_hat = conv2d(X)
#     l = (Y_hat - Y)**2
#     conv2d.zero_grad()
#     l.sum().backward()
#     conv2d.weight.data[:] -= lr * conv2d.weight.grad
#     if (i + 1) % 2 == 0:
#         print(f'epoch {i+1}, loss {l.sum():.3f}')

# print(conv2d.weight.data.reshape(1,2))
## tensor([[ 1.0047, -0.9768]]) 细心的人能够发现，学习到的卷积核权重非常接近我们之前定义的卷积核k

## 互相关和卷积
## 数学上面的卷积是需要反转的，但是深度学习中的卷积核是可学习的，所以不需要反转，两种的效果是一致的

## 特征映射和感受野
## 我们将卷积层的输出，我们就叫他是特征映射，本质上是对输入图像进行一次“特征提取 + 空间转换”
## 感受野：某一层的一个神经元，它的感受野是指，输入图像中，所有能影响这个神经元计算的区域大小
## 在单层卷积层的感受野来说，输出的神经元中，它的感受野就是卷积核的大小，在多层卷积中，网络越深，后面层的神经元，感受野就越大，能看到输入图像中更广阔的区域


## 填充和步幅
## 通过之前的学习，我们了解到，输出的大小与输入大小和卷积核的大小有关，公式是（nh - kh + 1） * (nw - kw + 1)
## 此外还能够影响的有填充和步幅，填充能够影响最后输出的图像的大小，保证减少信息丢失
## 步幅是卷积核在输入图像中移动的大小，能够减少无关信息的影响

## 我们通常添加pk行和ph列进行填充 则输出的形状就为 ：(nh - kh + ph + 1) * (nw - kw + pw + 1) 这意味着 输出的高度和宽度分别增加ph 和 pw
## 在许多情况下，我们通常设置 ph = kh -1 , pw = kw  -1 这样做的输入图像就和输出图像就是一致的
## 假设 kh是奇数，则我们就在高度的两侧填充 ph / 2 行

def comp_conv2d(conv2d,X):
    X = X.reshape((1,1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])
conv2d = nn.Conv2d(1,1,3,padding=1)
X = torch.rand(size= (8 ,8))

# print(comp_conv2d(conv2d,X).shape)

## 步幅
## 通过调整步幅的大小，我们可以跳过中间的位置，每次滑动多个元素
## 通常设置垂直步幅为 sh 垂直步幅为 sw
## 计算公式为 (（nh - kh + ph + sh）/ sh) * ((nw - kw + pw + sw) / sw)
## 若是 ph = kh - 1 和 pw = kw - 1
conv2d = nn.Conv2d(1,1,3,padding=1,stride=2)
# print(comp_conv2d(conv2d,X).shape)
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
# print(comp_conv2d(conv2d,X).shape)


## 多输入多输出通道
## 多输入通道之间互相关运算，就是对每个通道执行互相关操作，然后将结果相加

def corr2d_multi_in(X,K):
    ## X ：多通道输入 shape :(C_in , h, w)
    ## K : 多通道卷积核 shape: (C_in , k_h , k_w)
    out = 0
    for x,k in zip(X,K):
        out += corr2d(x,k)
    return out
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X,K))

## 多输出通道
def corr2d_multi_in_out(X,K):
    ## 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算
    ## 最后将所有的结果叠加到一起
    return torch.stack([corr2d_multi_in(X,k) for k in K] ,0)
## 多通道的本质，就是让卷积层用多组四维卷积核，同时提取多种不同的特征，输出多通道的特征图，
K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(corr2d_multi_in_out(X,K))

## 1 * 1卷积层
## 1 * 1卷积层 对图像的每个像素位置，单独做一次通道维度的线性变换，不改变空间大小，只改变通道数的通道维度全连接层
## 1 * 1卷积层的作用：1.通道数升降维，2，跨通道特征融合，3.大幅减少计算量
def corr2d_multi_in_out_1x1(X,K):
    c_i, h, w = X.shape ## 取出：输入通道数 ，高，宽
    c_o = K.shape[0] ## 取出：输出通道数
    X = X.reshape((c_i,h * w)) ##  把空间维度 H×W 拉平成一列！

    K = K.reshape((c_o,c_i)) ## 把 1x1卷积层 拉平成二维矩阵

    Y = torch.matmul(K,X) ## 矩阵cheng
    return Y.reshape(c_o,h,w)

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(Y1)
print(Y2)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6


## 6.5 汇聚层
## 汇聚层也就是池化层，它的作用是 在不改变通道数的前提下，缩小特征图的高和宽
## 池化层有两个目的，1.降低空间分辨率，扩大感受野，2.增强特征的平移不变性，降低位置敏感性
## 常见的池化操作有：最大池化（取区域内的最大值）,平均池化（取区域内的平均值）
## 池化层与卷积层的区别是：1，没有可学习的参数，都是固定操作，2.对位置的敏感性低，对小偏移不敏感








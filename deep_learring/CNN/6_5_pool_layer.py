## 6.5 汇聚层
## 汇聚层也就是池化层，它的作用是 在不改变通道数的前提下，缩小特征图的高和宽
## 池化层有两个目的，1.降低空间分辨率，扩大感受野，2.增强特征的平移不变性，降低位置敏感性
## 常见的池化操作有：最大池化（取区域内的最大值）,平均池化（取区域内的平均值）
## 池化层与卷积层的区别是：1，没有可学习的参数，都是固定操作，2.对位置的敏感性低，对小偏移不敏感
import torch
from torch import  nn

def pool2d (X,pool_size,mode = 'max'): ## 这里的 mode 是默认为 max ，但是也可以赋值为avg
    p_h,p_w =pool_size ## 取出池化窗口的高和宽
    Y =torch.zeros((X.shape[0]-p_h +1, X.shape[1]-p_w +1)) ## 计算输出大小，和卷积层无填充，步幅为1的公式一致
    ##  遍历 输出的每一个位置
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            ## 取出输入中对应池化窗口的区域
            window = X[i:i+p_h,j:j+p_w]
            if mode == 'max':
                Y[i,j] = window.max() ## 最大池化，取区域内最大值
            elif mode == 'avg':
                Y[i,j] = window.mean() ## 平均池化，取区域内平均值1
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X,(2,2)))
print(pool2d(X,(2,2),'avg'))

## 6.5.2. 填充和步幅
## 池化层也可以通过填充和步幅来改变形状，
## 下面，我们用深度学习框架中内置的二维最大汇聚层，来演示汇聚层中填充和步幅的使用。
X = torch.arange(16,dtype=torch.float).reshape(1,1,4,4)
print(X)
## 默认情况下，深度学习中的步幅和汇聚窗口大小相同，因此我们采用形状为 3x3 的汇聚窗口，在默认情况下，我们得到的步幅形状就为 3 x 3

pool2d = nn.MaxPool2d(3)
print(pool2d(X))
## 当然填充和步幅可以手动设置
pool2d = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))
## 我们可以设置任意的大小的矩形汇聚窗口，并分别设定填充和步幅的高度和宽度
pool2d = nn.MaxPool2d((2,3),stride = (2,3),padding = (0,1))
print(pool2d(X))

## 6.5.3. 多个通道
## 在处理多通道数据的时候，池化层在每个输入通道上面单独运算，而不像卷积层一样，在通道上面对输入进汇总，这意味着汇聚层的输出通道数和输入通道数相同
##  下面，我们将在通道维度上连结张量X和X + 1，以构建具有2个通道的输入。
X = torch.cat((X,X + 1),1)
print(X)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

## 小结
## 对于给定的输入数据，最大汇聚层会输出窗口内的最大平均值，平均汇聚层会输出窗口内的平均值
## 汇聚层的主要优点是减轻卷积层对于位置的过度敏感
## 我们可以设定汇聚层的填充和步幅
## 使用最大二维汇聚层以及大于1的步幅，可减少空间维度
## 汇聚层的输出通道数与输入通道数相同
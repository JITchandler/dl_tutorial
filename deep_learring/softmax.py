import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from IPython import display

import sys

def get_dataloader_workers():
    # Windows 必须用 0，否则报错
    return 0 if sys.platform.startswith('win') else 4

def load_data_fashion_mnist(batch_size, resize=None):
    # 数据转换：转成 tensor（自动归一化到 0~1）
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 下载加载数据
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )

    # 返回 训练迭代器 + 测试迭代器
    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers())
    return train_iter, test_iter

# 用法和 d2l 一模一样！
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

## 展平每一个图像，将它们视为长度为784的向量，因为我们的数据集有10个类别，所以网络输出维度为10
num_input = 28*28
num_output = 10

w = torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)
b = torch.zeros(num_output,requires_grad=True)

"实现softmax"
def softmax(x):
    x_exp = torch.exp(x) # 指数
    partition = x_exp.sum(1,keepdim=True)# 每行求和
    return x_exp / partition # 广播机制，得到概率 ，概率之和为1

def net(x):
    return softmax(torch.matmul(    x.reshape((-1,w.shape[0])),w) + b)

## 从预测概率中将真实标签对应的概率挑出来
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]

## 定义损失函数，交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)),y])
cross_entropy_loss = cross_entropy(y_hat, y)
print("交叉熵损失:",cross_entropy_loss)

## 计算模型精度正确率的核心参数
## 传入预测值y_hat和真实标签y，返回预测对了多少的样本

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    ### 返回正确的总数
    return float(cmp.type(y.dtype).sum())

print("精确度：",accuracy(y_hat, y)/len(y))
## 定义一个累加器类
class Accumulator:
    def __init__(self,n): ## 初始化n个0
        self.data = [0.0] * n

    def add(self,*arges): ## 累加
        self.data = [a + float(b) for a,b in zip(self.data,arges)]

    def reset(self): ## 清空归零
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx): ## 方便取数
        return self.data[idx]

def evaluate_accuracy(data_iter, net):
    metric = Accumulator(2)
    with torch.no_grad(): ## 关闭梯度，避免报错
        for x,y in data_iter:
            metric.add(accuracy(net(x),y),y.numel())
    return metric[0] / metric[1]
print("测试集准确率：", evaluate_accuracy(test_iter, net))

def train_epoch_ch3(net,train_iter,loss,updater): ## 模型，训练数据迭代器，损失函数，优化器
    if isinstance(net,torch.nn.Module): ## 如果模型是pytorch官方模型，就是设置为训练模式
        net.train()
    metric = Accumulator(3) ## 创建三个格子的累加器，存储总损失，正确预测的数量，总样本数量
    for x,y in train_iter:
        y_hat = net(x)
        l = loss(y_hat,y) ##模型预测，计算损失
        if isinstance(updater,torch.optim.Optimizer): ##   如果是pytorch优化器，
            updater.zero_grad() ## 梯度清零
            l.backward() ## 反向传播
            updater.step() ## 更新权重w和偏置 b
            metric.add(float(l.detach()) * len(y), ## 总损失
                       accuracy(y_hat,y), ## 正确数
                       y.size().numel() ## 总数
                       )
        else: ## 如果是自己手写的优化器
            l.sum().backward() ## 损失求和->反向传播
            updater(x.shape[0]) ## 调用手写更新参数
            metric.add(float(l.sum().detach()),accuracy(y_hat,y),y.numel())
    return metric[0] / metric[2] , metric[1] / metric[2] ## 返回 平均损失  = 总损失 / 总数  准确率 = 正确数 / 总数

## 定义一个在动画中绘制数据的实用程序类
class Animator:
    """在动画中绘制数据（无 d2l 纯版）"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 图例
        if legend is None:
            legend = []

        # 启用 SVG 清晰显示（替代 d2l.use_svg_display()）
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['savefig.dpi'] = 300

        # 创建画布
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]

        # 配置坐标轴（替代 d2l.set_axes）
        def config_axes():
            ax = self.axes[0]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            if legend:
                ax.legend(legend)
            ax.grid(True)

        self.config_axes = config_axes

        # 存数据
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """向图中添加数据点并动态更新"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n

        # 初始化数据列表
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]

        # 追加数据
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        # 清空画布重绘
        self.axes[0].cla()
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt)

        # 设置样式并显示
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

## 训练函数
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = Animator(xlabel = 'epoch',xlim = [1,num_epochs],ylim=[0.3,0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(test_iter,net)
        animator.add(epoch + 1,train_metrics + (test_acc,))
    train_loss,train_acc = train_metrics
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    with torch.no_grad():
        # 安全原位更新，不会破坏叶子节点的梯度
        w.sub_(lr * w.grad / batch_size)
        b.sub_(lr * b.grad / batch_size)

        # 梯度清零
        w.grad.zero_()
        b.grad.zero_()
num_epochs = 5
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)









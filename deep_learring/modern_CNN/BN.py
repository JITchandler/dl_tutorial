## 批量规范化（Batch Normalization，BN）
import time

## 核心作用：给神经网络中的每层的输入做标准化，解决神经网络训练难，收敛慢，梯度容易消失的问题
## 基本原理：对一个批次的数据，做两步处理
## 1.标准化：将数据缩放到均值为0，方差为1
## 2.缩放偏移，再用可学习参数微调，恢复网络表达能力

## 主要优点：
## 收敛快，降低对初始学习率，权重初始化的要求，一定程度上防止过拟合，缓解深层网络的梯度消失

## 简单使用：
## 通常加在，卷积/全连接层之后，激活函数之前
## 对于全连接层，作用在特征维
## 对于卷积层，作用在通道维


## 总结：批量归一化 固定小批量中的均值和方差，然后学习出合适的偏移和缩放
##  可以加速模型的收敛速度，一般不改变模型的精度

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # X:输入数据，gamma：缩放参数（可学习），beta：偏移参数（可学习），moving_mean：滑动平均均值（推理用）
    # moving_var：滑动平均方差（推理用），eps：防止除 0，momentum：滑动更新系数

    ## 判断是训练还是预测
    if not torch.is_grad_enabled():
        # 预测模式：直接用移动平均的均值方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        ## 训练模式：算当前的batch的均值方差
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:  ## 全连接层 BN
            mean = X.mean(dim=0)  ## 对batch 维度求平均
            var = ((X - mean) ** 2).mean(dim=0)  ## 形状：[batch, features]，每个特征自己算均值方差
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)  ##形状：[batch, channels, H, W]
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var

    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # ✅ 修复 1：__init__ 双下划线
    def __init__(self, num_features, num_dims):
        # ✅ 修复 2：super() 正确写法
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9
        )
        return Y


# ✅ 修复 3：去掉 num_dims= 关键字
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, 4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, 4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), BatchNorm(120, 2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, 2), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_iter = DataLoader(test_dataset, batch_size, shuffle=False)


def evaluate_accuracy(net, data_iter, device=None):
    if device == None and isinstance(net, nn.Module):
        device = next(net.parameters()).device

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def Init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(Init_weights)
    print("training on", device)
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer = []
    total_start = time.time()

    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(num_epochs):
        net.train()
        total_correct = 0
        total_loss = 0
        total_num = 0
        start_time = time.time()

        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += l.item() * X.shape[0]
                _, predicted = torch.max(y_hat, 1)
                total_correct += (predicted == y).sum().item()
                total_num += X.shape[0]

        train_loss = total_loss / total_num
        train_acc = total_correct / total_num
        test_acc = evaluate_accuracy(net, test_iter, device)

        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc * 100)
        test_acc_list.append(test_acc)

        epoch_time = time.time() - start_time
        timer.append(epoch_time)
        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'train loss {train_loss:.3f} | '
              f'train acc {train_acc:.3f} | '
              f'test acc {test_acc:.3f}')

    speed = total_num / (sum(timer) / num_epochs)
    print('-' * 50)
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{speed:.1f} examples/sec on {str(device)}')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, train_loss_list, 'r-o', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失变化')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, train_acc_list, 'g-o', label='训练准确率')
    plt.plot(epoch_list, test_acc_list, 'b-o', label='测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('准确率变化')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ✅ 修复 4：补上括号
train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
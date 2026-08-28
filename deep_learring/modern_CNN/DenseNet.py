##  稠密连接网络
## 与ResNet不同的是，DenseNet不是简单的逐层残差相加，而是每一层的输入 = 前面所有输出的拼接
## 关键模块
## 稠密块：Dense Block：块内各层稠密链接，通道数持续上涨
## 过渡层：Transition Layer：Dense Block 之间,用1 x 1卷积 + 池化降通道，降尺寸，控制参数和计算量
## 增长率k：每层只输出K个特征通道，控制通道膨胀速度
import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

## 稠密块体
## DenseNet使用了ResNet改良版的“批量规范化、激活和卷积”架构,我们先实现一下这个架构
def conv_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1),
    )
##一个稠密块由多个卷积块组成，每个卷积块使用相同数量的输出通道。 然而，在前向传播中，我们将每个卷积块的输入和输出在通道维上连结。
class DenseBlock(nn.Module):
    def __init__(self,num_convs,input_channels,num_channels):
        super(DenseBlock,self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(
                conv_block(
                    num_channels * i + input_channels,
                    num_channels
                )
            )
        self.net = nn.Sequential(*layers)
    def forward(self,X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X,Y),1)
        return X
blk = DenseBlock(2,3,10)
X = torch.randn(4,3,8,8)
Y = blk(X)
print(Y.shape)
##  过渡层
## 稠密层的使用会增加通道数，过多使用则会增加模型的复杂度，我们采用过渡层来控制模型的通道数
## 使用 1 x 1卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低模型复杂度

def transition_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=1),
        # nn.AvgPool2d(kernel_size=2,stride=2),
    )
blk = transition_block(23,10)
print(blk(Y).shape)

## DenseNet模型
## 构造DenseNet模型,DenseNet首先使用同ResNet一样的单卷积层和最大汇聚层。
b1 = nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(64),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
)

num_channels,growth_rate = 64,32 ## 当前通道数 = 64，增长率 k =32
num_convs_in_dense_block = [4,4,4,4] # 4个稠密块，每个里面有4个卷积层
blks = [] # 存放所有网络层
for i,num_convs in enumerate(num_convs_in_dense_block):
    # 1. 创建一个稠密块，加入网络
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))

    # 2.更新通道数，加上这个稠密块新增的通道，
    num_channels += num_convs * growth_rate

    #3 .如果不是最后一个块，加过度层， 把通道数减半
    if i != len(num_convs_in_dense_block)-1:
        blks.append(transition_block(num_channels,num_channels // 2))
        num_channels = num_channels // 2
## 与ResNet类似，最后连上全局汇聚层和全连接层来输出结果
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))

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

lr, num_epochs, batch_size = 0.1, 10, 256
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_iter = DataLoader(test_dataset, batch_size, shuffle=False)
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



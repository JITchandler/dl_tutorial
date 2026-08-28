## 残差网络 ResNet
## 传统卷积神经网络（VGG、AlexNet）层数加深后，准确率反而下降，这不是过拟合，而是梯度消失 / 梯度爆炸 + 网络退化：
## 梯度消失：反向传播时，梯度不断乘小于 1 的数，越往前层梯度趋近于 0，参数无法更新。
## 网络退化：深层网络难以拟合恒等映射，简单任务都学不好。
import time

## 核心思想：残差块 ：F(x) = H(x) - x 最终输出 H(x) = F(x) + x , 简单理解：主干分支学习“增量变化” ，捷径分支学习传递输入
## 主干分支：堆叠卷积，BN，激活函数，学习特征变化
## 捷径分支：直接跳过几层卷积，把输入x原样传输到输出端

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides =1):
      super().__init__()
      self.conv1 = nn.Conv2d(input_channels,num_channels,
                             kernel_size=3,stride=strides,
                             padding=1)
      self.conv2 = nn.Conv2d(num_channels,num_channels,
                             kernel_size = 3,
                             padding =1)
      if use_1x1conv:
          self.conv3 = nn.Conv2d(input_channels, num_channels,
                                 kernel_size=1,stride=strides,
                               )
      else:
          self.conv3 = None
      self.bn1 = nn.BatchNorm2d(num_channels)
      self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
blk = Residual(3,3)
X = torch.rand(4,3,6,6)
Y =blk(X) ## 残差块没有改变形状
print(Y.shape)
## torch.Size([4, 6, 3, 3])
## ResNet 模型
## ResNet的前两层跟之前介绍的GoogLeNet中的一样：在输出通道为 64，步幅为2的 7x7 卷积层之后，
## 接步幅为 2的3x3的最大汇聚层，不同之处在于ResNet每个卷积层后增加了批量规范化层
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
## 帮你堆叠N个残差块，组成ResNet的一大层（Stage）
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = [] # 核心判断：只有【不是第一个stage】 + 【是这一层的第一个残差块】
    for i in range(num_residuals): ###
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            ## 剩下的残差块：尺寸，通道都不变
            blk.append(Residual(num_channels, num_channels))
    return blk ## 返回一整组残差块
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
lr, num_epochs, batch_size = 0.05, 10, 256
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


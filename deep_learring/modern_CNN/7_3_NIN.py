import time

import torch
from matplotlib import pyplot as plt
from torch import  nn
from torchvision import datasets ,transforms
from torch.utils.data import DataLoader

## NiN 网络中的网络，
## 提出了NiN块的组合： Conv(3×3) + ReLU → Conv(1×1) + ReLU → Conv(1×1) + ReLU
## 其中采用了 1x1 的卷积层，不改变特征图尺寸，只在通道维度做特征融合，非线性变换，等价于在每个像素位置中套了一个全连接层

## 全局平均池化层：
## 传统CNN：最后使用大参数全连接层分类，容易过拟合，参数爆炸
## NIN：去掉所有的全连接层，对最后一层的特征图采用全局平均池化，每个通道对应一个分类，整张特征图求均值，直接输出结果
## 处理流程：输入图像 → 堆叠多组 NiN 块 + 最大池化 → 最后一层 NiN 块 → 全局平均池化 → 输出分类

## 总结:
##  NIN块使用卷积层加了两个 1 x 1的卷积层，后者对每个像素增加了非线性
##  NIN块使用全局平均池化层来替代原来的全连接层，不容易过拟合，更少的参数


def nin_block(in_channels, out_channels, kernel_size,stride,padding ):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding=padding),
        nn.ReLU(),
        ## 两个 1 x 1
        nn.Conv2d(out_channels, out_channels, kernel_size = 1),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels, kernel_size = 1),nn.ReLU()
    )

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    ## 标签类别数是 10
    nin_block(384,10,3,1,1),
    nn.AdaptiveAvgPool2d((1,1)),
    ## 将四维的输出转成二维的输出，其形状为 （批量大小，10）
    nn.Flatten()
)
X= torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

## 7.3.3 模型训练
lr, num_epochs, batch_size = 0.1, 10, 128
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 关键：和 resize=224 效果一样
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)
## 加载Fashion_MNIST 数据集
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
train_iter = DataLoader(train_dataset,batch_size = batch_size,shuffle = True)
test_iter = DataLoader(test_dataset,batch_size = batch_size,shuffle = False)
def evaluate_accuracy(net,data_iter, device = None):
    ## 如果没有指定device,自动使用模型所在的设备
    if device == None and isinstance(net,nn.Module):
        device = next(net.parameters()).device

    net.eval() ## 评估模式
    correct = 0
    total = 0
    with torch.no_grad(): ## 禁用梯度
        for X,y in data_iter:
            # 数据搬到设备上
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            # 预测类别（取输出最大值的索引）
            _, predicted = torch.max(output.data, 1)
            # 统计总数 + 正确数
            total += y.size(0)
            correct += (predicted == y).sum().item()
    ## 返回准确率
    return 100 * correct / total

def train_ch6 (net,train_iter,test_iter,num_epochs,lr,device):

    ## 1,初始化权重
    def Init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(Init_weights)
    print("training on",device)
    net.to(device)

    ## 2.优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(),lr = lr)
    loss = nn.CrossEntropyLoss()
    ## 计时
    timer = []
    total_start = time.time()

    # ===================== 【新增】用于绘图的列表 =====================
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    ## 开始训练
    for epoch in range(num_epochs):
        net.train() ## 训练模式
        total_correct = 0
        total_loss = 0
        total_num = 0
        start_time = time.time()

        for i,(X,y) in enumerate(train_iter):
            # 数据搬到GPU
            X ,y =X.to(device),y.to(device)

            ## 前向 + 反向传播 + 参数更新
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()

            ## 累计损失和准确率
            with torch.no_grad():
                total_loss += l.item() * X.shape[0]
                _,predicted = torch.max(y_hat, 1)
                total_correct += (predicted == y).sum().item()
                total_num += X.shape[0]
        ## 一轮训练结束
        train_loss = total_loss / total_num
        train_acc = total_correct / total_num
        test_acc = evaluate_accuracy(net,test_iter,device)

        # ===================== 【新增】保存数据 =====================
        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc * 100)
        test_acc_list.append(test_acc)

        epoch_time = time.time() - start_time
        timer.append(epoch_time)
        # 打印进度（替代原来的动画）
        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'train loss {train_loss:.3f} | '
              f'train acc {train_acc:.3f} | '
              f'test acc {test_acc:.3f}')
    ## 最后总结
    speed = total_num / (sum(timer) / num_epochs)
    print('-' * 50)
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{speed:.1f} examples/sec on {str(device)}')

    # ===================== 【新增】绘图 =====================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, train_loss_list, 'r-o', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失变化')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
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
train_ch6(net, train_iter, test_iter, num_epochs, lr,device)


import time

import torch
from matplotlib import pyplot as plt
from torch import  nn
from torchvision import datasets ,transforms
from torch.utils.data import DataLoader
## VGG
## 经过上次的AlexNet的提出，所呈现的趋势是，我采用的卷积是否更深，更大从而就能得到更好的精度呢
##VGG是在原来AlexNet的基础上面，提出了VGG块的概念，是由卷积层所组成的块
## VGG块是由若干个 3X3 卷积,填充为1，有m个通道 和 2x2的最大池化层 所组成的，

## VGG架构，多个VGG块后接全连接层，不同次数的重复块得到不同的架构
## 总结：
## VGG使用可重复的卷积块来构建深度卷积神经网络
##不同的卷积块和超参数可以得到不同的复杂度的变种

## 定义VGG块函数，内有三个参数，分别是卷积层的数量，输入通道数，输出通道数
def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels ## 下一层的输入通道等于 = 上一层的输入通道
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

## 7.2.2 VGG网络
## VGG块中采用conv_arch 超参数来指定每个VGG块的中的卷积层的个数和输出通道数
## 原始VGG网络有5个卷积块，前2个里面各有一个卷积块，后3个里面各有两个卷积块
##第一个模块里面有 64个输出通道，每个后续模块将输出模块数翻倍，直到该数字达到 512
## 由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11
conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
def vgg(conv_arch):
    conv_blks =[]
    in_channels = 1 ## 用于单通道的灰度图
    ## 卷积层部分： 搭建所有卷积快
    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels = out_channels
    ## 把所有卷积块 + 全连接层拼在一起
    return nn.Sequential(
        *conv_blks, ## 所有卷积块
        nn.Flatten(), ## 展平，特征图变成一维向量
        ## 全连接部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
net = vgg(conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

## 7.2.3 模型训练
lr, num_epochs, batch_size = 0.05, 10, 128
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
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ch6(net, train_iter, test_iter, num_epochs, lr,device)


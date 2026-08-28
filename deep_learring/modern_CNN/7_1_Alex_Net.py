## AlexNet 是在2012年推出的卷积神经网络结构，可以理解为是在原来 LeNet的基础上面 优化了，可以应对更复杂的数据
## AlexNet 采用了8层卷积神经网络，并以很大的优势赢得了2012年ImageNet图像识别挑战赛。
## AlexNet 与 LeNet的差异：
# AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层。
# AlexNet使用ReLU而不是sigmoid作为其激活函数。
import time

import torch
from matplotlib import pyplot as plt
from torch import  nn
from torchvision import datasets ,transforms
from torch.utils.data import DataLoader

from deep_learring.linear_regression import num_epochs

net = nn.Sequential(
    ## 这里使用一个11*11的更大窗口来捕捉对象。
    ## 同时，步幅为4，以减少输出的高度和宽度
    ## 另外，输出的通道数目远大于LetNet
    nn.Conv2d(1,96,kernel_size =11,stride = 4,padding = 1), nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3,stride = 2),
    ## 减小卷积核窗口，通过填充为2来使得输入和输出的高和宽一致，且增大输出窗口
    nn.Conv2d(96,256,kernel_size = 5,padding = 2),nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3,stride = 2),
    ## 使用三个连续的卷积层和较小的卷积窗口
    ## 除了最后的卷积层，输出通道的数量进一步增加
    ## 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256,384,kernel_size = 3,padding = 1),nn.ReLU(),
    nn.Conv2d(384,384,kernel_size = 3,padding =1),nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    ## 这里，全连接层的输出数量是lenet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400,4096),nn.ReLU(),
    nn.Dropout(p = 0.5),
    nn.Linear(4096,4096),nn.ReLU(),
    ## 最后的输出层，由于这里采用的是Fashion-MNIST,所以用类别为10
    nn.Linear(4096,10)
)

X = torch.randn(1, 1, 224, 224)
# for layer in net:
#     X=layer(X)
#     print(layer.__class__.__name__,'output shape:\t',X.shape)
## 读取数据集
batch_size = 128
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

lr ,num_epochs = 0.01,10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ch6(net, train_iter, test_iter, num_epochs, lr,device)
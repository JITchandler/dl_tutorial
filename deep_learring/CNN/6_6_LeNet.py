## 卷积神经网络
## 这里我们讨论 LeNet 卷积神经网络
## LeNet 卷积神经网络 是由两大模块组成
## 卷积编码器 ：包含 2个卷积层和 2个池化层，负责特征提取
## 全连接密集块：包含3个全连接层，负责最终分类
## 每块卷积块中的基本单位是 5x5 卷积层，sigmoid激活函数和平均汇聚层，
import time
## 其中卷积层将输入映射到多个二维输出，通常会增加通道数的数量，
## 每个池化操作会通过空间下采样，将维数减少
import torch
from torch import  nn
from torchvision import datasets ,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

net = nn.Sequential(
    # 第1层：卷积
    # 输入：1×28×28
    # 输出：6×28×28 （padding=2保证尺寸不变）
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
    # 第2层：平均池化（尺寸减半）
    # 输出：6×14×14
    nn.AvgPool2d(kernel_size=2,stride=2),
    # 第3层：卷积
    # 输出：16×10×10
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
    # 第4层：平均池化（尺寸减半）
    # 输出：16×5×5
    nn.AvgPool2d(kernel_size=2,stride=2),

    # 展平成一维向量 16*5*5 = 400
    nn.Flatten(), ## 负责将四维[256, 16, 5, 5]变为二维[256, 400]
   ## 全连接层
    nn.Linear(16 * 5 * 5, 120),nn.Sigmoid(),  #把 400 维特征 → 120 维特征
    nn.Linear(120, 84),nn.Sigmoid(),# 再压缩为 84 维特征
    nn.Linear(84, 10) ## 最后输出10类 # 最后输出 10 维
)

## 将一个 28x 28的单通道（黑白）图像通过图像通过LeNet,打印每一层打印输出的形状，来见检查模型。
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


# 6.6 模型训练
batch_size = 256
transform = transforms.Compose(
    [
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


lr, num_epochs = 0.9, 10
# 自动选择 GPU / CPU，完全替代 d2l.try_gpu()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
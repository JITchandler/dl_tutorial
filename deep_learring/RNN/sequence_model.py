## 序列模型

## 处理时序 / 有序数据（文本、语音、时间序列、股价），数据前后存在依赖关系，输入是长度不定的有序序列，核心是利用上文信息预测当前 / 后续输出。
## 特点：输入有序、上下文相关、长短可变。
## 传统的序列模型有自回归模型，马可夫链，如下所示：
## 自回归模型：是基于历史的数据进行拟合，从而达到对未来数据的预测，多用于金融、气象单变量时序预测，但是只能捕捉线性规律
## 自回归模型有两种策略：1，不选择从第一个数据开始到最近的数据，来拟合数据。而是选择满足某个长度为t的时间的跨度
##                   2.第二种是保留一些对过去数据的总结H（t）,从而就会需要两种模型，X`（t） = P(X(t)| H(t)) 以及公式 H（t） = g(H(t-1),X(t-1))
## 上面的第二种的自回归模型的近似法，我们采用的是近似精确的，我们就说序列满足马可夫条件

## 训练，我们先生成一些数据，使用正弦函数和一些可加性噪声来生成序列数据。
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
# 1. 生成时间序列总长度：1000个时间步
T= 1000
# 2. 生成时间点：1,2,...,1000
time = torch.arange(1,T + 1,dtype = torch.float32)
# 3. 生成带噪声的正弦序列（核心：序列数据）
x = torch.sin(0.01 * time) + torch.normal(0,0.02,(T,))

# plt.figure(figsize=(6, 3))
# plt.plot(time, x)
# plt.xlabel('time')
# plt.ylabel('x')
# plt.xlim(1, 1000)
# plt.show()

# 2. 构造特征&标签
tau = 4 ## 嵌入窗口大小，用连续前tau个历史数据作为输入特征，预测下一个时刻
features = torch.zeros((T - tau,tau)) ## 每4个点能够预测1个点，所以能够构造出1000 - 4 = 996 组样本，每一组样本的长度是4，所以 features 形状 = (996, 4)
for i in range(tau):
    features[:,i] =x [i:T - tau + i]## 这里的循环就是分组
labels = x[tau:].reshape((-1,1)) ##x[tau:] = x[4:]，这些就是每一组要预测的真实值，形状：(996, 1)


batch_size , n_train = 16, 600
train_fea = features[:n_train]
train_lab = labels[:n_train]
# 用原生DataLoader代替d2l
train_dataset = TensorDataset(train_fea, train_lab)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

## 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

## 一个简单的多层感知机
def get_net():
    net = nn.Sequential(
        nn.Linear(4,10),
        nn.ReLU(),
        nn.Linear(10,1),
    )
    net.apply(init_weights)
    return net
## 平方损失，注意MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')
def evaluate_loss(net, data_iter, loss):
    net.eval()  # 评估模式，关闭dropout/bn
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad(): # 不计算梯度，节省显存
        for X, y in data_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            total_loss += l.sum().item()
            total_samples += l.numel()
    net.train()  # 切回训练模式
    return total_loss / total_samples

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            pred = net(X)
            l = loss(pred, y)
            l.sum().backward()
            trainer.step()
        # 改用自定义evaluate_loss
        train_loss = evaluate_loss(net, train_iter, loss)
        print(f'epoch {epoch + 1}, loss: {train_loss:f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 一步预测
onestep_preds = net(features)

# 原生 matplotlib 绘图（完全替代 d2l.plot）
plt.figure(figsize=(6, 3))
plt.plot(time.numpy(), x.detach().numpy(), label='data')
plt.plot(time[tau:].numpy(), onestep_preds.detach().numpy(), label='1-step preds')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim(1, 1000)
plt.legend()
plt.show()

## 单步预测：只用真实历史数据，预测下一个点，因为没有用预测值当输入，没有误差累积
## 多步预测，从某一时刻开始，只用预测出来的值，继续往后预测，特点：误差不断累计，越往后面越不准
# 多步递归预测
multistep_preds = torch.zeros(T) # 1. 创建一个长度为 T=1000 的空数组，存放多步预测结果
multistep_preds[: n_train + tau] = x[: n_train + tau]  # 前半部分用真实数据
# 2. 前 n_train+tau 个位置，直接填入【真实数据】
#    意思是：前 600+4=604 个点是真实已知的，从这里开始往后预测


# 3. 从第 604 个点开始，一直到最后 1000，开始【递归多步预测】
for i in range(n_train + tau, T):
    # 用过去 tau 个预测值，继续预测下一个
    # 4. 核心：
    #    取 multistep_preds[i-4 : i] → 前4个值
    #    重点：这4个可能已经是【预测值】了！
    #    送入网络，预测第 i 个点
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))

# 绘图：原始数据 + 单步预测 + 多步预测
plt.figure(figsize=(6, 3))
plt.plot(time.numpy(), x.detach().numpy(), label='data', color='blue')
plt.plot(time[tau:].numpy(), onestep_preds.detach().numpy(), label='1-step preds', color='orange')
plt.plot(time[n_train + tau:].numpy(), multistep_preds[n_train + tau:].detach().numpy(), label='multistep preds', color='green')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim(1, 1000)
plt.legend()
plt.show()
max_steps = 64

# 构造多步预测特征矩阵
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

# 要展示的预测步长
steps = (1, 4, 16, 64)

# 绘图（纯 matplotlib）
plt.figure(figsize=(6, 3))
for i in steps:
    plt.plot(
        time[tau + i - 1 : T - max_steps + i].numpy(),
        features[:, tau + i - 1].detach().numpy(),
        label=f'{i}-step preds'
    )
plt.xlabel('time')
plt.ylabel('x')
plt.xlim(5, 1000)
plt.legend()
plt.show()

## 小结：
## 1.内插法（在现有观测值之间进行估计）和外推法（对超出已知观测进预测）在实际的实践中差别很大，因此，对于所拥有的序列数据，在训练的时候要尊重其时间顺序，最好不要基于未来的数据进行训练
## 2.序列模型目前流行的模型是自回归模型和隐变量自回归模型
## 3。对于时间是向前推进的因果模型，正向估计通常比反向估计更容易。
## 4.在进行多步预测的时候，会造成极大的误差和预测质量的下降。









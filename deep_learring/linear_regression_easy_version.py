import gluon
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
true_w = torch.tensor([2., -3.4])
true_b = 4.2

## 生成数据
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    # 生成特征 X
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算标签 y
    y = torch.matmul(X, w) + b
    # 加入小噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))  # 把标签变成列向量

features,labels = synthetic_data(true_w,true_b,1000)

def load_array(data_arrays,batch_size,is_training=True):
    dataset = TensorDataset(*data_arrays) ##把特征和标签数据打包起来
    return DataLoader(dataset, batch_size, shuffle=is_training) ##将打包好的数据按批次，随机打乱，喂给模型

batch_size = 10
data_iter = load_array((features,labels),batch_size)
## 初始化模型参数，设置线性回归模型中的权重和偏置
net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.normal_(0)

## 定义损失函数 MSELoss 也可以称为是平方范数
loss  = nn.MSELoss()

## 定义优化算法，小批量随机梯度下降算法
trainer = torch.optim.SGD(net.parameters(), lr=0.01)

## 训练过程
num_epochs = 3
for epoch in range(num_epochs):
    for x , y in data_iter:
        l = loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l: f}')






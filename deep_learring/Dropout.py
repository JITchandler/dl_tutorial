## Dropout是方式在训练神经网络中，防止训练过拟合
## 它的核心思想是通过使得神经网络中的某些神经元不工作，从而强迫神经网络中的每一部分都需要独立的提取特征
import torch
from torch import  nn

def dropout_layer(X,dropout):
    assert 0 <= dropout <= 1 ## 首先声明 dropout是在 0 - 1 之间
    if dropout == 0:  ## 丢弃率为 0 ,则直接返回 X
        return X
    if dropout == 1: ## 丢弃率为 1 , 则返回 与 X 相同的 0 张良
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    ## 生成和X形状相同的1，值在0 -1之间的随机数，如果 生成的随机数大于 dropout 则保留，返回 true，转化成整数为 1
    ## 反之 丢弃，转化成整数为 0
    return mask * X / (1.0- dropout)
## mask = 0 则返回 0 ， mask = 1 则返回X，(1.0- dropout) 则是保证数值的范围是相同的，因为丢弃了一部分，所以平均值减少，所以通过(1.0- dropout)把数值拉回原来的尺度
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1 , dropout2 = 0.2,0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,is_training):
        super(Net,self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshpe((-1 ,self.num_inputs))))
        if self.training == True:
            H1 = dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2,dropout2)
        out  = self.lin3(H2)
        return out
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)




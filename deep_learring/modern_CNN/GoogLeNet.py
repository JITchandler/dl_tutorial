import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception,self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4,kernel_size=1)

    def forward(self,x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),1)

# ===================== 修复 1：第一层 stride=2 =====================
b1 = nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7, stride=2,padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# ===================== 修复 2：输入尺寸改为 224 =====================
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# ===================== 修复 3：学习率下调为 0.03 =====================
lr, num_epochs, batch_size = 0.03, 10, 32

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

train_dataset = datasets.FashionMNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, transform=transform, download=True
)

train_iter = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_iter = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

def evaluate_accuracy(net,data_iter, device=None):
    if device == None and isinstance(net,nn.Module):
        device = next(net.parameters()).device

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X,y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def train_ch6 (net,train_iter,test_iter,num_epochs,lr,device):
    def Init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(Init_weights)
    print("training on",device)
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
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

        for i,(X,y) in enumerate(train_iter):
            X ,y =X.to(device),y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += l.item() * X.shape[0]
                _,predicted = torch.max(y_hat, 1)
                total_correct += (predicted == y).sum().item()
                total_num += X.shape[0]

        train_loss = total_loss / total_num
        train_acc = total_correct / total_num
        test_acc = evaluate_accuracy(net,test_iter,device)

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
train_ch6(net, train_iter, test_iter, num_epochs, lr,device)
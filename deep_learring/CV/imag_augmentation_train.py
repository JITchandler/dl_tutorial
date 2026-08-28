
## 采用图像增广来进行训练，使用CIFAR-10数据集
import ssl

import torch
from torch import nn

ssl._create_default_https_context = ssl._create_unverified_context
import torchvision
from d2l import torch as d2l

all_images = torchvision.datasets.CIFAR10(root='./data', train=True,download=True)
## 对数据集中的前 32 个训练图像如下所示
d2l.show_images([all_images[i][0] for i in range(32)],4,8,scale= 0.8)
d2l.plt.show()

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

## 定义一个辅助函数，便于读取图像和应用图像增广，Gluon数据集提供的transform_first函数将图像增广应用于每个训练样本的第一个元素

def get_devices():
    """获取可用设备，如果没有GPU则使用CPU"""
    if torch.cuda.is_available():
        return d2l.try_all_gpus()
    else:
        return ['cpu']
def load_cifar10(is_train,augs,batch_size):
    dataset = torchvision.datasets.CIFAR10(root = './data', train=is_train, download=True, transform=augs)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=d2l.get_dataloader_workers())

    return dataloader

def train_batch_ch13(net,X,y,loss,trainer,devices):

    """用多GPU进行小批量训练"""
    if isinstance(X,list):
        """微调BERT中所需"""
        X = [x.to(devices[0])  for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred,y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    trian_acc_sum = d2l.accuracy(pred,y)
    return train_loss_sum,trian_acc_sum

def train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs,devices = d2l.try_all_gpus()):
    """用多GPU进行小批量训练"""
    timer,num_batches = d2l.Timer(),len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net,device_ids=devices).to(devices[0])

    # 如果有多GPU，使用DataParallel
    if len(devices) > 1 and devices[0] != 'cpu':
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    else:
        # 单GPU或CPU
        device = devices[0] if devices else 'cpu'
        net = net.to(device)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4) # 存储 [损失和，准确数和，样本数，元素数]

        for i,(features,labels) in enumerate(train_iter):
            timer.start()
            l,acc = train_batch_ch13(net,features,labels,loss,trainer,devices)
            metric.add(l,acc,labels.shape[0],labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2],  # 平均损失
                              metric[1] / metric[3],  # 训练准确率
                              None))  # 测试准确率暂不更新
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

batch_size,devices,net = 256,d2l.try_all_gpus(),d2l.resnet18(10,3)

def init_weights(m):
    if type(m) in [nn.Linear,nn.Conv2d]:
        nn.init.xavier_normal_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs,test_augs,net,lr = 0.001):
    train_iter = load_cifar10(True,train_augs,batch_size)
    test_iter = load_cifar10(False,test_augs,batch_size)
    loss = nn.CrossEntropyLoss(reduction = 'none')
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net,train_iter,test_iter,loss,trainer,10,devices)

train_with_data_aug(train_augs,test_augs,net)

## 总结：
## 1.图像增广基于现有的训练数据随机生成随机图像，来提高图像泛化能力
## 2.为了得到确切的结果，我们通常对训练数据进图像增广，在预测过程中不使用带随机操作的图像增广
## 3.深度学习框架提供了不同的图像增广方法，这些方法可以同时使用
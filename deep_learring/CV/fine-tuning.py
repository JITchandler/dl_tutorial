## fine-tuning 微调

## 先用大型的数据集来预先训练一个模型，之后再用自己的小数据来继续训练这个已经学会基础特征的网络模型
## 从头训练：随机初始化权重，从零学习；微调：权重初始化为预训练权重，在此基础上更新参数

## 两种常见的微调策略
## 一：特征提取
## 1。冻结预训练主干网络中的所有参数，只替换最后的分类头，只是训练新增的几层
## 2.优点：快，不容易拟合，缺点：模型表单能力上线低
## 二：完整微调：解冻主干网络中的部分参数或者是全部参数，连同新的分类头一起训练
## 载入预训练权重，替换最后一层输出层，使用更小学习率的训练网络


## 热狗识别
import ssl

import gluon
from torch.nn import init

ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch
import torchvision
from matplotlib import pyplot as plt
from mpmath.libmp import normalize
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
print(data_dir)
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

## 展示前8张正类样本和最后的父类样本
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i -1][0] for i in range(8)]
def show_images(imgs,num_rows,num_cols,titles=None,scale = 1.5):
    ## 计算整张图的大小
    figsize = (num_cols*scale,num_rows*scale)
    _, axs = plt.subplots(num_rows,num_cols,figsize=figsize)
    ## 把多维的画板拉平成一维列表，方面后续的循环画图
    axs = axs.flatten()
    ## 便利每一个画板，每一张图片，一对一的画
    for i,(ax,img) in enumerate(zip(axs,imgs)):
        # 如果是 Tensor，先转 numpy
        if hasattr(img, 'numpy'):
            img = img.numpy()
        ax.imshow(img)
        # 隐藏坐标轴
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axs
show_images(hotdogs + not_hotdogs, 2,8,scale=1.4)

## 图像预处理
normalize = torchvision.transforms.Normalize(
    [0.485,0.456,0.406],[0.229,0.224,0.225]
)

## 训练增广
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224), ## 随机裁剪并且缩放
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize,
])
## 测试增广
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256,256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize,
])

## 定义和初始化模型

## 我们使用在ImageNet数据集上预训练的ResNet-18作为源模型。 在这里，我们指定pretrained=True以自动下载预训练的模型参数。

pretrained_net = torchvision.models.resnet18(pretrained=True)

print(pretrained_net.fc)

## 之后，我们构建一个新的神经网络，它的定义方式和原来的预训练源模型的定义方式相同，只是在输出层不一样

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc= nn.Linear(finetune_net.fc.in_features,2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

## 微调模型
def train_fine_tuning(net,learning_rate,batch_size = 128,num_epochs = 5,param_groups = True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction = 'none')
    if param_groups: ## 分层微调，取出所有的参数，排除fc.weight,fc.bias（最后的分类头）
        params_1x = [param for name,param in net.named_parameters()
                    if name not in ['fc.weight','fc.bias']]
        trainer = torch.optim.SGD(
            [
                {'params': params_1x}, ## 主干特征：使用默认 lr
                {'params': net.fc.parameters(), ## 最后的全连接分类层
                 'lr': learning_rate * 10 ## 学习率放大 10 倍
                 }
            ],
            lr = learning_rate,weight_decay = 0.001
        )
    else: ## 全局微调，所有参数统一同一个学习率
        trainer = torch.optim.SGD(
            net.parameters(),
            lr = learning_rate,
            weight_decay= 0.001
        )
    d2l.train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs,devices)

## train_fine_tuning(finetune_net,5e-5)

## 为了比较，我们定义一个相同的模型，但是将其所有模型参数初始化为随机值，由于整个模型需要从头训练，因此我们需要使用更大的学习率
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_groups=False)

## 小结
## 1.迁移学习将从源数据集中学到的知识迁移到目标数据集，微调是迁移学习的常见技巧
## 2.除了输出层之外，目标模型从源模型中复制所有模型设计及其参数，并且根据目标数据集对这些参数进行微调，
    ## 但是 目标模型的输出层需要从头开始训练
## 3.通常，微调参数使用较小的学习率，而从头训练输出层可以使用更大的学习率

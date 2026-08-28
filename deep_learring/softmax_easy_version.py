import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
def get_dataloader_workers():
    return 0 if sys.platform.startswith('win') else 4

# 加载 FashionMNIST （和 d2l.load_data_fashion_mnist 完全一样）
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 加载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )

    # 返回数据迭代器
    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers())
    return train_iter, test_iter

# 用法和原来一模一样！
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

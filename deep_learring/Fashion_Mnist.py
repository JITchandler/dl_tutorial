import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from torch.utils.data import dataloader
from torchvision import transforms

from deep_learring.linear_regression import batch_size

trans = transforms.ToTensor()
## 训练数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                transform=trans,
                                                download=True)
##测试数据集
mnist_test = torchvision.datasets.FashionMNIST(root='./data',
                                               train=False,
                                               transform=trans,
                                               download=True)
# print(len(mnist_train))
# print(len(mnist_test))

def get_fashion_mnist_lables(labels):
    "返回fashion_mnist_数据集的文本标签"
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


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

x , y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
show_images(x.reshape(18,28,28),2,9,titles=get_fashion_mnist_lables(y))

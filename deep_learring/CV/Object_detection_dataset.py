## 目标检测数据集
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

def read_bananas(is_train = True):
    """读取香蕉检测数据中的图像和标签"""
    ## 自动下载香蕉数据并且解压，返回解压后的根目录路径
    data_dir = d2l.download_extract('banana-detection')

    ## 拼接csv文件路径
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val','label.csv')

    ## 使用pandas来读取csv，并且将文件名 img_name 设置成索引
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')

    images, targets = [],[]
    ## img_name :图片名字 如0.png target :一行标签数据 [label,xmin,ymin,xmax,ymax]

    ## torchvision.io.read_image() 读取图片，直接返回 torch 张量，格式 (C,H,W)，像素值范围 0~255

    for img_name,target in csv_data.iterrows():
        images.append(torchvision.io.read_image(os.path.join(data_dir,'bananas_train' if is_train else 'bananas_val','images',f'{img_name}')))
        ## 将每一行的标签 [label,xmin,ymin,xmax,ymax]转成列表存入targets
        targets.append(list(target))

    ## 图片数据归一化
    return images,torch.tensor(targets).unsqueeze(1) / 256

class BananaDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self,is_train):
        self.features,self.labels = read_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
                                                   is_train else f' validation examples'))
    def __getitem__(self, idx):
        return (self.features[idx].float(),self.labels[idx])

    def __len__(self):
        return len(self.features)

## 得到两个数据迭代器
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    ## 训练集
    train_iter = torch.utils.data.DataLoader(
        BananaDataset(is_train=True),
        batch_size,
        shuffle=True, )

    ## 验证集
    val_iter = torch.utils.data.DataLoader(
        BananaDataset(is_train=False),
        batch_size,
    )
    return train_iter,val_iter

batch_size,edge_size = 32,256
train_iter,val_iter = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape)
print(batch[1].shape)

imgs = (batch[0][0:10].permute(0,2,3,1)) / 256
axes = d2l.show_images(imgs,2,5,scale = 2)
for ax , label in zip (axes,batch[1][0:10]):
    d2l.show_bboxes(ax,[label[0][1:5] * edge_size],colors=['w'])

d2l.plt.show()















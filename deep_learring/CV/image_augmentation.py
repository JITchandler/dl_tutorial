
## 图像增广，对现有的训练图片做随机变换，生成新的，但是标签一样的图片用来扩充训练数据集
## 例如：将 猫 图片 进行旋转，裁剪，放大，缩小等
## 核心关键点：图片变了，但是分类标签不变
import torchvision
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
## imshow（）运行之后，图片只会存放在内存中，不会显示图片
d2l.plt.imshow(img)
d2l.plt.show()

## 定义 辅助apply，此函数在输入图像img上多次运行图像增广方法aug并显示所有结果
def apply(img,aug,num_rows =2,num_cols = 4,scale = 1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y,num_rows,num_cols,scale =scale)
## 翻转和裁剪
## 左右翻转图像通常不会改变对象的类别，这是最早且最广泛使用图像增广的方法之一
## 我们使用transforms模块来创建RandomFlipLeftRight实例
apply(img,torchvision.transforms.RandomHorizontalFlip())
d2l.plt.show()
## 上下翻转图像，一般比较少用，上下翻转不会妨碍识别
## 创建一个RandomFlipTopBottom实例，使得图像各有百分之五十的几率向上或者向下
apply(img,torchvision.transforms.RandomVerticalFlip())
d2l.plt.show()

## 如果图像的目标并不是在正中央，则我们需要对图片进行裁剪，使得物体以不同的比例出现在图像的不同位置
## 这也可以降低模型对于目标位置的敏感性

shape_aug = torchvision.transforms.RandomResizedCrop(
    (200,200),
    scale=(0.1,1.0),
    ratio = (0.5,2)
)
apply(img,shape_aug)
d2l.plt.show()

## 改变颜色
## 我们可以改变图像颜色的四个方面：亮度，对比度，饱和度和色调
apply(img,torchvision.transforms.ColorJitter(0.5,0,0,0))
d2l.plt.show()

## 我们可以改变图像的色调
color_aug = torchvision.transforms.ColorJitter(0,0,0,0.5)
apply(img, color_aug)
d2l.plt.show()

## 在实践中，我们将结合多种图像增广方法，比如，我们可以通过使用一个Compose实例来综合上面定义的不同的图像增广方法
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug,
    shape_aug
])
apply(img,augs)
d2l.plt.show()
## objection detection 目标检测

## 在图像中，可能有很多我们需要关注的物体，我们不仅需要分辨这个物体的类别，还要知道这个物体的位置，这样的任务，我们就称为是目标检测

## 目标检测的应用广泛，比如在自动驾驶中，汽车需要识别在行驶过程中的所遇到的物体，从而实现自动的避让等操作。
import torch
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open("../img/catdog.jpg")
# d2l.plt.imshow(img)
# d2l.plt.show()

## 边界框
## 我们经常用边界框来描述对象的空间位置，边界框是矩形的，由矩形，左上角的以及右下角的x和y决定，
## 另一种常用的边界框表示法是边界框中心的(x,y)轴坐标以及框的宽度和高度
## 这里设置两个函数用来转换

def box_corner_to_center(boxes):
    """"从（左上，右下）转换到（中间，宽度，高度）"""
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis = -1)
    return boxes


def box_center_to_corners(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx,cy,w,h = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    boxes = torch.stack((x1,y1,x2,y2), axis = -1)
    return boxes
## 我们根据坐标信息来定义，猫狗的边界框，图中坐标的原点是图像的左上角，向右的方向为x轴的正方向，向下的方向为y轴的正方向
# bbox是边界框的英文缩写
## 它这边是手动调节的，感觉有点low
dog_bbox, cat_bbox = [187, 8.1, 408.1, 274.4], [69.4, 62.7, 200, 278.8]

## 我们可以通过转换两次来验证边界框函数的正确性

boxes = torch.tensor((dog_bbox,cat_bbox))
print(box_center_to_corners(box_corner_to_center(boxes)) == boxes)

## 我们可以通过在图像上画出边界框，从而看出检查是否准确，首先我们定义一个函数，来将边界框转化为matplotlib的形式
def bbox_to_rect(bbox,color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy = (bbox[0],bbox[1]),
        width = bbox[2] - bbox[0],
        height = bbox[3] - bbox[1],
        fill = False,
        edgecolor = color,
        linewidth = 2
    )
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
d2l.plt.show()

## 小结：
## 目标检测不仅可以识别图像中所有感兴趣的物体，还能识别他们的位置，由矩形边界框表示
## 我们可以在两种常用的边界框表示（中间，宽度，高度）和（左上，右下）坐标之间进行转换。
## 多尺度目标检测

import torch
from d2l import torch as d2l

img = d2l.plt.imread('../img/catdog.jpg')
h,w = img.shape[:2]
print(h,w)

def display_anchors(fmap_w,fmap_h,s):
    d2l.set_figsize()
    fmap = torch.zeros(1,10,fmap_h,fmap_w)
    ## 生成锚框，特征图 fmap,尺寸 s 长宽高的比ratio
    anchors = d2l.multibox_prior(fmap,sizes = s,ratios=[1,2,0.5])
    ## 把归一化锚框坐标 ->映射回原像素坐标(w,h是原图宽高)
    bbox_scale = torch.tensor((w,h,w,h))
    ## 画原图 + 叠加锚框
    d2l.show_bboxes(d2l.plt.imshow(img).axes,anchors[0]*bbox_scale)
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
d2l.plt.show()
d2l.plt.clf()
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
d2l.plt.show()
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
d2l.plt.show()
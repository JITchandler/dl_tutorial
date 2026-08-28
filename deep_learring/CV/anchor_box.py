## 锚框
from itertools import repeat

## 目标检测算法会在输入图像中采集大量的区域，并且判断该区域内是否有我们感兴趣的东西
## 如果有，则调整预测目标的真实边界框
## 其中有一种方法：以每个像素为中心，生成多个缩放比和宽高比不同的边界框
import torch
from d2l import torch as d2l
from mpmath import rect

torch.set_printoptions(2)


## 生成多个锚框
## 同一像素为中心的锚框的数量是 n + m -1 ，对于整个输入图像，将共生成 wh(n + m - 1) 个锚框

## 生成锚框的方法在下面的multibox_prior函数中实现
## 指定输入图像，尺寸列表和宽高比列表，然后将此函数返回所有的锚框

def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    ## 严格对应课本公式 n+m-1，每个像素生成多少个锚框
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    ## 像素中点归一化计算（关键：算出所有像素的中心坐标）
    offset_h, offset_w = 0.5, 0.5
    ## 计算步长
    step_h = 1.0 / in_height
    step_w = 1.0 / in_width
    ## 将像素坐标归一化
    center_h = (torch.arange(in_height, device=device) + offset_h) * step_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * step_w
    ## 生成整张图片所有像素点（y,x）网络坐标
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    ## 把二维网络平摊成一维列表，所有像素中心按照顺序排成一列
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    ## 根据课本公式计算 每一种锚框的 宽 w，高 h
    ## 计算锚框的宽和锚框的高
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] / torch.sqrt(ratio_tensor[1:])))

    ## 计算锚框相对中心点的偏移量（半宽，半高）
    ## 锚框坐标公式
    ## Xmin = center - w / 2
    ## Ymin = center - h / 2
    ## Xmax = center + w / 2
    ## Ymax = center + h / 2

    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    ## 像素中心点坐标复制匹配锚框数量

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    ## 中心点 + 偏移 = 最终锚框坐标
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])

## 可以看出返回锚框Y的形状是（批量大小，锚框的数量，4）
print(Y.shape)

## 我们可以将锚框变量Y的形状更改（图像高度，图像宽度，以同一像素为中心的锚框的数量，4）后，我们可以
## 获得以指定像素的位置为中心的所有锚框
## 将会显示4个元素，左上角的 x, y坐标，和 右下角的 x, y坐标，输出中两个坐标分别表示了图像的宽度和高度

boxes = Y.reshape(h, w, 5, 4)
print(boxes[250, 250, 0, :])


## 为了显示图像中某个元素为中心的所有锚框，定义下面的函数来在图像上绘制多个边界框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""

    ##  统一入参格式，做兼容处理
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    ## 统一格式化 labels colors
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    ## 循环绘制每一个方框
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        ## 绘制框上面的文字标签
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))



d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)


# show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
#             ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
#              's=0.75, r=0.5'])
# d2l.plt.show()

## 我们使用交并比来衡量锚框和真实边界框之间，以及不同锚框之间的相似度
## 给定两个锚框或者边界框的列表，采用以下函数计算这两个列表之间的成对的交并比

def box_iou(boxes1, boxes2):
    """"计算两个锚框或者是边界框列表中成对的交互比"""

    ## 计算单个框的面积
    ## box[] : xmin,ymin,xmax,ymax
    box_area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    areas1 = box_area(boxes1)  ## shape[N] ## A 组里面所框的面积
    areas2 = box_area(boxes2)  ## shape[M] ## B 组里面所框的面积
    ## inter_upperlefts,inter_lowright ,inter的形状：
    ## （boxes1的数量，boxes2的数量）

    ## 计算交集坐标，左上方有交集，则就取两个框左上角作业的最大值
    ##            右下角有交集，则就去两个框右下角作业的最小值

    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    ## 计算重叠的宽高，没有交集就清零
    ## 重叠宽 = 重叠右x - 重叠左x
    ## 重叠高 = 重叠下y - 重叠上y
    ## 如果两个框压根不相交，则计算所得是负数，则直接为零
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    ## 重叠面积 = 重叠宽 * 重叠高
    ## 并集面积 = 两块面积相加 - 重叠面积
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_area = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_area


## 接下来，我们需要锚框的 类别 class 和 偏移量offset 使得锚框能够更加接近真实的边界值
## 我们通过矩阵的方式来将真实边界框分配给锚框
## 在矩阵中找到交并比最大的位置 ij,之后将这一列的所有元素全部删除，直到所有的列都被删除
## 以下函数实现这个功能 ，最后返回长度 = 锚框的数量

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于 第i列和第j列的元素X_iJ是锚框i和真实值边界j的IOU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    ## 初始化映射表，一开始锚框默认是 -1，然后将符合条件的改成对应真实框编号
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    max_ious, indices = torch.max(jaccard, dim=1)
    ## 取出锚框的最大值和最大值的坐标
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j

    ## 这样的话，就会出现一种情况，某个真实框的所有锚框的IOU都不到 0.5 那就不会有物体取拟合，训练就会失败
    ## 所以强制遍历每一个真实框，全局找当前IOU最大的框锚框分给他
    clo_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)

    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()  ## 算出这个最大值属于第几号真实框
        anc_idx = (max_idx // num_gt_boxes).long()  ## 算出这个最大值属于第几号锚框
        anchors_bbox_map[anc_idx] = box_idx  ## 将这个锚框强制的分配给真实值,将原来的值 -1 给覆盖掉
        ## 上面的这个真实框付给锚框之后，将这一列的所有真实框都赋值为 -1
        jaccard[:, box_idx] = clo_discard
        jaccard[anc_idx, :] = row_discard  ##该锚框整行赋值 -1 ，这个锚框不会在匹配其他真实框
    return anchors_bbox_map


## 标记类别和偏移量
## 要知道水平方向偏移和竖直方向偏移，宽度缩放偏移和高度缩放偏移的公式计算

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    ## 两角坐标 -> 中心坐标
    c_anc = d2l.box_corner_to_center(anchors)  # 输入：xmin,ymin,xmax,ymax
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)  # 输出（中心点xy,框宽x,框高h）

    ## 计算中心点偏移 offset_xy(dx,dy)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    ## 计算宽高对数偏移 offset_wh(dw,dh)
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    ## 拼接两个偏移
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def mulitbox_target(anchors, labels):
    batch_size = labels.shape[0]
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    ## 循环遍历batch里面每一张图片单独处理
    for i in range(batch_size):
        label = labels[i, :, :]
        # 1.锚框和当前图片的真实值做匹配，得到映射数组
        ## anchors_bbox_map 长度 = 锚框数量
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)

        ## 2.生成回归掩码
        ## unsqueeze(-1) 升维 -> [num_anchors,1]
        ## repeat(1,4) 横向复制4列 [num_anchors,4]
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        ## 3.初始化分类标签，匹配框坐标数组(默认全0)
        ## class_labels:初始为0 ，0代表背景
        ## asigned_bb:初始为0，存放每个锚框对应的真实框坐标，背景框这里的数值没有用
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        ## 4,给匹配成功的锚框赋值真实的类别，真实框坐标
        indices_true = torch.nonzero(anchors_bbox_map >= 0)  # 拿到所有正样本（匹配）锚框下标
        bb_idx = anchors_bbox_map[indices_true]  ## 每个正锚框对应真实框的编号
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        ## 5. 计算偏移，掩码屏蔽背景偏移，通过计算,背景锚框全部请零，只有正样本保留真实偏移标签
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask

        ## 6.把单张图数据存进列表,将二维数据展平成一维，方便后续整体拼接计算损失
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    ## 循环结束，把所有图片数据拼接成完成的batch张量
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)  ## 最后打包一整批次数据返回给模型计算损失


ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
# d2l.plt.show()

## 下面为锚框和真实值边界框样本添加一个维度
label = mulitbox_target(anchors, ground_truth.unsqueeze(dim=0))
print(label[2])
## 返回第一个元素包含了为每个锚框标记的四个偏移值，注意，负类锚框的偏移值被标记为零
print(label[1])
print(label[0])

## 使用非极大值来抑制边界框
## 之前我们是通过由锚框算出真实框的偏移标签，
## 一个预测好的边界框则是根据其中某个带有预测偏移量的锚框而生成，下面的函数是通过锚框 + 网络预测出来的偏移 推算出最终测框

def offset_inverse(anchors,offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)

    ## 公式反推，还原预测框中心坐标
    pred_bbox_xy = (offset_preds[:,:2] * anc[:,2:] / 10) + anc[:,:2]
    ## 还原预测框宽高
    pred_bbox_wh = torch.exp(offset_preds[:,2:] / 5) * anc[:,2:]

    ## 转回左上角-右下角格式
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

## 当有多个重叠独很高的检测框的时候，利用NMS函数，按照可信度从高到低排序，每轮保留最高分框
## 删掉和它冲得太多的重复框，循环结束，剔除重复检测

def nms(boxes,scores,iou_threshold):
    """对预测边界框的置信独进行排序"""
    B = torch.argsort(scores,dim=- 1,descending=True) ## 返回分数从高到低之后，框的下表位置
    keep = [] ## 保留预测边界框的指标
    while B.numel() > 0: ## 只要列表B中还有元素，就一直循环
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i,:].reshape(-1,4),boxes[B[1:],:].reshape(-1,4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep,device = boxes.device)

## 下面采用multibox_detection函数将非极大值抑制应用于预测边界框

def multibox_detection(cls_probs,offset_preds,anchors,nms_threshold = 0.5,pos_threshold =0.009999999):
    ## cls_probs :[批次，类别数，锚框总数]，每个锚框属于每个类别的概率 第0类是背景
    ## offset_preds[批次，锚框 * 4]: 模型预测出来的xy-wh偏移量
    ## anchors 预先生成好的锚框的总数
    ## nms_threshold  NMS的重叠阈值，超过阈值就判定是同一个物体，直接删除
    ## pos_threshold 置信度门槛，分数低于该值就直接判定为背景
    device,batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_class,num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = [] ## 用来保存每张图片最后的检测结果

    for i in range(batch_size):
        cls_prob , offset_pred = cls_probs[i],offset_preds[i].reshape(-1,4)
        conf,class_id = torch.max(cls_prob[1:],0)
        predicted_bb = offset_inverse(anchors,offset_pred) ## 换算出真实像素坐标的预测框
        keep = nms(predicted_bb,conf,nms_threshold) ## keep 保留筛选之后留下来的，没有重复重叠的锚框的下标
        ## 找出被NMS筛掉的锚框non_keep

        all_idx = torch.arange(num_anchors,dtype = torch.long, device=device)
        combined = torch.cat((keep,all_idx))
        uniques,counts = combined.unique(return_counts = True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep,non_keep))
        class_id[non_keep] = -1 ## 被NM筛掉的框直接设置为类别为 -1
        ## 按照新排序，重排类别，置信度，预测框坐标
        class_id = class_id[all_id_sorted]
        conf,predicted_bb = conf[all_id_sorted],predicted_bb[all_id_sorted]
        ## 低分锚框强制设置为背景
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]

        ## 拼接一条检测信息
        pred_info = torch.cat((class_id.unsqueeze(1),
                              conf.unsqueeze(1),
                              predicted_bb),dim = 1)
        out.append(pred_info)
    return torch.stack(out)

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
##d2l.plt.show()
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)

d2l.plt.clf()  # 清空之前累积的图形
fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
d2l.plt.show()




















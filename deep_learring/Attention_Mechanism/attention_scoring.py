## 注意力评分函数
import math
import torch
from matplotlib import pyplot as plt
from torch import nn

## 掩蔽softmax函数操作
## 在进行将数据纳入注意力汇聚中时，某些文本序列因为长度不一致，导致在处理中添加了没有意义的特殊词元，掩蔽softmax操作就是为了将这些词元置为0
def show_heatmaps(matrices,xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    ## 获取子图行列的布局
    num_rows,num_cols = matrices.shape[0], matrices.shape[1]
    ## 创建画布与子图网络
    fig , axes = plt.subplots(num_rows , num_cols, figsize=(figsize[0]*num_cols, figsize[1]*num_rows),
                             sharex=True, sharey=True, squeeze=False)
    pcm = None # 保存最后一张图用于色条
    for i,(row_axes,row_matrices) in enumerate(zip(axes,matrices)):
         for j,(ax,matrix) in enumerate(zip(row_axes,row_matrices)):
             # tensor转 numpy,脱离计算图
            data = matrix.detach().cpu().numpy()
            pcm = ax.imshow(data, cmap=cmap)
             # 仅最后一行显示X标签
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            # 仅第一列显示y标签
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles is not None:
                ax.set_title(titles[j])
    # 全局颜色条：
    fig.colorbar(pcm,ax=axes,shrink=0.6)
    # 展示图像
    plt.show()
def sequence_mask(X,valid_len,value = 0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen),dtype = torch.float32,device = X.device)[None,:] < valid_len[:,None]
    X[~mask] = value
    return X

def masked_softmax(X,valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    # 情况1：不给有效长度，不需要掩码，直接普通softmax
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1)
    else:
        ## X 的原始三维形状为：(batch,query_num,seq_len)
        if valid_lens.dim() == 1:
            shape = X.shape
        # 分支1：valid_lens是1维 [batch_len1, batch_len2,...]
        # 例：batch=2，query_num=3，valid_lens=[2,3] → [2,2,2,3,3,3
            valid_lens = torch.repeat_interleave(valid_lens,shape[1])

        # 分支2：valid_lens是2维 (batch, query_num)
        else:
            ## 直接摊平成一维，和上面的格式统一
            valid_lens = valid_lens.reshape(-1)
    # 1. X压成2维：(batch*query_num, seq_len)
    # 2. sequence_mask：把每条里超过valid_lens的位置填 -1e6（极小负数）
    # 极大负数经过softmax后概率≈0，实现屏蔽填充位
    X = sequence_mask(X.reshape(-1,shape[-1]),valid_lens,value = -1e6)
    return nn.functional.softmax(X.reshape(shape),dim=-1)
## 经过掩蔽softmax的操作之后，超过有效长度的值被掩蔽为0
print(masked_softmax(torch.rand(2,2,4),torch.tensor([2,3])))

## 10.3.2 加性注意力
## 点积注意力只能用在 Q ,K,维度完全相等的情况，加性注意力专门解决Q ,K 向量长度不一样的情况

class AdditiveAttention(nn.Module):
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super(AdditiveAttention,self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size,num_hiddens,bias = False)
        self.W_q = nn.Linear(query_size,num_hiddens,bias = False)
        self.W_v = nn.Linear(num_hiddens,1,bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values,valid_lens):
        queries , keys = self.W_q(queries),self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        # queries.unsqueeze(2)：新增第 2 维 → (batch, 查询数, 1, h)
        # keys.unsqueeze(1)：新增第 1 维 → (batch, 1, 键值数, h)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # self.W_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores,valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights),values)
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
print(attention(queries,keys,values,valid_lens))

show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

## 10.3.3 缩放点积注意力
## 缩放点积注意力用矩阵乘法快速算QK的相似度，除以维度平方根稳住softmax梯度
class DotProductAttention(nn.Module):
    def __init__(self,dropout,**kwargs):
        super(DotProductAttention,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self,queries,keys,values,valid_lens = None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries,keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries,keys,values,valid_lens))

show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')


## 总结：
## 将注意力汇聚的输出计算可以作为值的加权平均，选择不同的注意力评分函数会带来不同的注意力汇聚操作。
## 当查询和键是不同长度的矢量时，可以使用可加性注意力评分函数。当它们的长度相同时，使用缩放的“点－积”注意力评分函数的计算效率更高。



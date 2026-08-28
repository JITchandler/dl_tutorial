## 自注意力
## 让序列中的每个词都能和序列中的所有词建立关联，自动捕捉全局依赖
## 计算Q和K的相似度，得到注意力分数
## 使用softmax归一化，得到权重
## 用权重加权求和所有V,得到当前词融合全局信息的输出

import math
import torch
from matplotlib import pyplot as plt
from torch import nn

from deep_learring.Attention_Mechanism.attention_prompt import show_heatmaps

## 基于多头注意力对一个张量进完成自注意力的计算
num_hiddens , num_heads = 100,5
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
## 缩放点积注意力用矩阵乘法快速算QK的相似度，除以维度平方根稳住softmax梯度

def transpose_qkv(X, num_heads):
    # X: (batch, n, hidden)
    # 拆分多头，变换维度适配并行多头计算
    batch, n, hidden = X.shape
    head_dim = hidden // num_heads
    X = X.reshape(batch, n, num_heads, head_dim)
    X = X.permute(0, 2, 1, 3)  # (batch, heads, n, head_dim)
    return X.reshape(-1, n, head_dim)  # (batch*heads, n, head_dim)

def transpose_output(X, num_heads):
    # 逆操作，把分开的多头还原拼接
    multi_batch, n, head_dim = X.shape
    batch = multi_batch // num_heads
    X = X.reshape(batch, num_heads, n, head_dim)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(batch, n, num_heads * head_dim)
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
class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias = False,**kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)## 多头拼接后的融合层

    def forward(self, queries, keys, values, valid_lens):
        queries =  transpose_qkv(self.W_q(queries),self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,repeats = self.num_heads,dim = 0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries,keys,values,valid_lens)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens).shape)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self,num_hiddens,dropout,max_lens =1000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(dropout)
        ## 构建核心公式，创建一个足够长的P
        self.P = torch.zeros((1,max_lens,num_hiddens))
        X = torch.arange(max_lens,dtype = torch.float32).reshape(-1,1) / torch.pow(1000,torch.arange(0,num_hiddens,2,dtype = torch.float32) / num_hiddens)
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)

    def forward(self,X):
        # 截取和输入序列等长的位置编码，移到和X相同设备
        X = X + self.P[:,:X.shape[1],:].to(X.device)
        return self.dropout(X)
# 超参
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()

# 输入全0向量，只保留位置编码部分
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]

# 取出第6~9四个维度：P[0, :, 6], P[0, :, 7], P[0, :, 8], P[0, :, 9]
positions = torch.arange(num_steps).numpy()
feats = P[0, :, 6:10].T.numpy()  # shape [4, num_steps]
labels = [f"Col {d}" for d in range(6, 10)]

# 绘图
plt.figure(figsize=(6, 2.5))
for i, feat in enumerate(feats):
    plt.plot(positions, feat, label=labels[i])

plt.xlabel("Row (position)")
plt.legend()
plt.tight_layout()
plt.show()
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')

## 小结
# 在自注意力中，查询，键和值都来自同一组输入
## 卷积神经网络和自注意力都有并行计算的优势，而且自注意力的最大路径长度最短，但是因为其计算复杂度是关于序列长度的二次方，所以在很长序列中的计算会非常慢
## 为了使用序列的顺序信息，可以通过在输入表示中添加位置编码，来注入绝对的或者相对的位置信息


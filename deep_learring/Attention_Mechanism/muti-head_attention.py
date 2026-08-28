## 多头注意力机制

## 核心思想：把单头注意力机制复制多份并行计算，每一头独立捕捉不同类型的语义关联，最后进行拼接融合

## 三步流程：
## 1.分头投影：将输入的 Q、K、V 分别线性变换，切分成 h 组小维度的 Qᵢ、Kᵢ、Vᵢ（h 就是头数，常见 8 头）。

## 2.分头算自注意力，每一头单独执行标准自注意力，各自得到一组加权特征
## 头1：侧重语法指代（代词对应名词）
## 头2：侧重局部相邻词语搭配
## 头3：侧重长距离逻辑关系
## 不同头捕捉不一样的关联模式

## 3.拼接 + 融合
## 把所有头输出的向量拼在一起，再做一次线性变换压缩维度，得到最终头注意力输出
import math
import torch
from torch import nn
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

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(X,Y,Y,valid_lens).shape)




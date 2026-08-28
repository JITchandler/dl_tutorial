import math

import torch
from torch import  nn
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
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.attention(X, X, X, valid_lens)
        X = self.addnorm1(X, Y)
        Y = self.ffn(X)
        X = self.addnorm2(X, Y)
        return X

def get_tokens_and_segments(tokens_a, tokens_b = None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens,segments

class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,max_len = 1000,key_size = 768,query_size = 768,value_size = 768,**kwargs):
        super(BERTEncoder,self).__init__(**kwargs)

        ## 三种嵌入层（BERT标志性三嵌入）
        self.tokens_embedding = nn.Embedding(vocab_size,num_hiddens) ## 将输入的单词映射为向量，存储词语本身语义
        self.segment_embedding = nn.Embedding(2,num_hiddens) ## 区分两个句子

        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}",EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
        ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))\
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1,max_len,num_hiddens))


    def forward(self,tokens,segments,valid_lens):
        """在以下代码中，X的形状保持不变，（批量大小，最大序列长度，num_hiddens）"""
        ## 1. 词嵌入 + 段落嵌入相加
        X = self.tokens_embedding(tokens) + self.segment_embedding(segments)
        ## 2. 叠加位置嵌入。截取到当前序列长度
        X = X + self.pos_embedding.data[:,:X.shape[1],:]

        ## 3. 逐层经过所有Transformer Encoder 块
        for blk in self.blks:
            X = blk(X,valid_lens)
        return X
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
print(encoded_X.shape)

## 预训练任务
## 包括两个任务: 掩蔽语言模型和下一句预测
## MLM 任务 将被MASK盖住的位置的向量，预测会原来的单词ID，就是BERT预训练的核心人物掩码语言模型

class MaskLM(nn.Module):
    def __init__(self,vocab_size,num_hiddens,num_inputs = 768,**kwargs):
        super(MaskLM,self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens), # 升维变换
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size), # 输出词表每个单词得分
        )

    def forward(self,X,pred_positions):
        ## 1.展平所有mask位置
        num_pred_positions = pred_positions.shape[1] ## 每个句子里面有多少的mask词
        pred_positions = pred_positions.reshape(-1) # [2,3] -> [0,1,2,0,1,2]

        ## 2.生成对应batch索引
        batch_size  = X.shape[0]
        batch_idx = torch.arange(0,batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx,num_pred_positions)

        ## 取出所有mask位置的向量
        masked_X = X[batch_idx,pred_positions]
        masked_X = masked_X.reshape((batch_size,num_pred_positions,-1))

        mlm_Y_hat = self.mlp(masked_X)
        ## 输出shape =[batch,mask个数，vocab_size]
        # 每个mask位置，输出此表中每个单词的预测logits
        return mlm_Y_hat
mlm = MaskLM(vocab_size,num_hiddens)
mlm_postions= torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X,mlm_postions)
print(mlm_Y_hat.shape)

## 计算交叉损失
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1,vocab_size)),mlm_Y.reshape(-1))
print(mlm_l.shape)

## NSP 下一句预测头
## BERT 判断两端文本是不是原文连续的上下句
## 输出二分类：0 = 是连续句子，1 = 不是连续句子

class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self,num_inputs,**kwargs):
        super(NextSentencePred,self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs,2)

    def forward(self,X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
print(nsp_Y_hat.shape)

nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
print(nsp_l.shape)

class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
         super(BERTModel,self).__init__()
         # 1.核心编码器: 三嵌入 + 多层 Transformer Encoder
         self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                    dropout, max_len=max_len, key_size=key_size,
                                    query_size=query_size, value_size=value_size)
         # 2. NSP专用隐藏层，对CLS对向量变换
         self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                     nn.Tanh())
         # 3，MLM掩码语言模型头，预测mask单词
         self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
         # 4.NSP二分类头，判断两句是否连续
         self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


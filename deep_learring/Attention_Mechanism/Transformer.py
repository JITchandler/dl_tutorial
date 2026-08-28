import math
import os
import zipfile
from timeit import Timer
import requests
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import time
## 基于位置的前馈神经网络
## 基于位置的前馈神经网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机
## 所以称前馈网络是基于位置的原因
## 带有注意力机制解码器
##解码器
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)
    ## 新增一个init——state函数，用于将编码器的输出，转化成编码后的状态
    def _init_state(self,enc_outputs,*args):
        raise NotImplementedError

    ## 为了逐个生成长度可变的词元序列，解码器在每个时间步，都会将输入和编码后的状态映射成当前时间步的输出词元
    def forward(self,X,state):
        raise NotImplementedError
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self,**kwargs):
        super(AttentionDecoder,self).__init__(**kwargs)
    @property
    def attention_weights(self):
        raise NotImplementedError
class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)
    def forward(self,x,*args):
        raise NotImplementedError
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
        shape = X.shape
        ## X 的原始三维形状为：(batch,query_num,seq_len)
        if valid_lens.dim() == 1:
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
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self,ffn_num_input,ffn_num_hiddens,ffn_num_outputs,**kwargs):
        super(PositionWiseFFN,self).__init__(**kwargs)

        self.dense1 = nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens,ffn_num_outputs)
    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))

ffn = PositionWiseFFN(4,4,8)
x = torch.ones((2,3,4))
out = ffn(x)
print(out.shape)
## 残差链接和层规范化
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# 在训练模式下计算X的均值和方差
print('layer norm:', ln(X), '\nbatch norm:', bn(X))

## 可以使用残差链接和层规范化来实现AddNorm类，暂退法被作为正则化方法使用
class AddNorm(nn.Module):
    """残差链接之后使用层规范化"""
    def __init__(self,normalized_shape,dropout,**kwargs):
        super(AddNorm,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self,X,Y):
        return self.ln(self.dropout(Y) + X)

add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)
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
## 编码器
class EncoderBlock(nn.Module):
    """transform编码器"""
    def __init__(self,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,use_bias = False,**kwargs):
        super(EncoderBlock,self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.ffn = PositionWiseFFN(ffn_num_input,ffn_num_hiddens,ffn_num_input)
        self.addnorm2 = AddNorm(norm_shape,dropout)

    def forward(self,X,valid_lens):
        Y = self.addnorm1(X,self.attention(X,X,X,valid_lens))
        return self.addnorm2(Y,self.ffn(Y))
X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
print(encoder_blk(X, valid_lens).shape)
class TransformerEncoder(Encoder):
    """transformer编码器"""
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias = False,**kwargs):
        super(TransformerEncoder,self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                EncoderBlock(key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,use_bias))
    def forward(self, X, valid_lens,*args):
        ## 因为位置编码的值在 -1 和 1 之间
        ## 因此嵌入值乘以嵌入维度的平方进行缩放
        ## 然后再与位置编码相加
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i,blk in enumerate(self.blks):
            X = blk(X,valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)

## 解码器
## DecoderBlock类中实现的层中包含三个子层，掩码解码器自注意力，“编码器-解码器”注意力和基于位置的前馈网络
## 掩码解码器子注意力是加了遮挡掩码，防止生成时偷看未生成的词
## “编码器-解码器”注意力用来读取编码器的原文信息
## 前馈网络：对一个字的向量单独做非线性变换，先升维，再降维，不改变序列长度，不改变特征维度，最后走残差相加

class DecoderBlock(nn.Module):
    """解码器中的第i块"""
    def __init__(self,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ff_num_hiddens,num_heads,dropout,i,**kwargs):
        super(DecoderBlock,self).__init__(**kwargs)
        self.i = i # 标记当前是第几层的解码器，用来对应缓存cache
        ## 1.第一层，解码器掩码多头自注意力（只能看已经生成的词）
        self.attention1 = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout)
        self.addnorm1 = AddNorm(norm_shape,dropout) # 残差 + LN
        ## 第二层，编码器-解码器交叉注意力（独属于Decoder）
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        ## 3.第三层，基于位置的前馈网络FFN
        self.ffn = PositionWiseFFN(ffn_num_input,num_hiddens,num_hiddens) ## ...
        self.addnorm3 =AddNorm(norm_shape,dropout)

    def forward(self,X,state):
        ## X 是当前解码器的输入向量 （batch,num_step,d_model） model是模型的基础维度
        ## state 是三元组缓存容器[enc_outputs,enc_valid_lens,cache] 编码器的最终输出，有效长度，token缓存
        enc_outputs,enc_valid_lens = state[0],state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i] , X),axis = 1)
        ## 更新缓存，保留所有已经生成的tokens
        state[2][self.i] = key_values

        ## 生成掩码dec_valid_lens，实现单向遮挡
        if self.training:
            batch_size,num_steps,_ = X.shape
            dec_valid_lens = torch.arange(1,num_steps + 1,device=X.device).repeat(batch_size,1)
        else:
            dec_valid_lens = None
        # 自注意力
        # 第一层 掩码自注意力 attention1
        X2 = self.attention1(X,key_values,key_values,dec_valid_lens)
        Y = self.addnorm1(X, X2)
        ## 编码器-解码器注意力
        ## enc_outputs的开头，(batch_size,num_steps,num_hiddens)

        ## 第二层 交叉注意力 attention2
        Y2 = self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z = self.addnorm2(Y, Y2)

        ## 第三层 逐位置 FFN
        return self.addnorm3(Z,self.ffn(Z)),state
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state)[0].shape)

class TransformerDecoder(AttentionDecoder):
        def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,**kwargs):
            super(TransformerDecoder,self).__init__(**kwargs)
            self.num_hiddens = num_hiddens
            self.num_layers = num_layers
            self.embedding = nn.Embedding(vocab_size,num_hiddens)
            self.pos_encoding = PositionalEncoding(num_hiddens,dropout)
            self.blks = nn.Sequential()
            for i in range(num_layers):
                self.blks.add_module("block" + str(i),
                            DecoderBlock(key_size, query_size, value_size, num_hiddens,
                            norm_shape, ffn_num_input, ffn_num_hiddens,
                            num_heads, dropout, i))
            self.dense = nn.Linear(num_hiddens,vocab_size)
        def init_state(self,enc_outputs,enc_valid_lens,*args):
            return [enc_outputs,enc_valid_lens,[None] * self.num_layers]

        def forward(self, X, state):
            # 1. 词嵌入 + 缩放 + 位置编码
            X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
            # 2. 开辟空间保存注意力权重（用于后续可视化）
            self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
            # 3. 逐层循环所有DecoderBlock
            for i, blk in enumerate(self.blks):
                X, state = blk(X, state)
                # 保存两类注意力权重
                # _attention_weights[0]：解码器掩码自注意力（attention1）
                self._attention_weights[0][i] = blk.attention1.attention.attention_weights
                # _attention_weights[1]：编码器-解码器交叉注意力（attention2）
                self._attention_weights[1][i] = blk.attention2.attention.attention_weights
            # 4. 最后一层输出映射到词表，得到每个token预测得分
            return self.dense(X), state

        @property
        def attention_weights(self):
            return self._attention_weights

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
FILE_NAME = "fra-eng.zip"
FILE_URL = DATA_URL + FILE_NAME

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def stop(self):
        return time.time() - self.start_time
def preprocess_nmt(text):
    """预处理"""
    def no_space(char,prev_char):
        return char in set(',.!?') and prev_char !=''
    ## 使用空格替换不间断空格
    ## 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    ## 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char,text[i - 1]) else char for i,char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text,num_examples =None):
    source = []
    target= []
    for i ,line in enumerate(text.split("\n")):
        if num_examples and i > num_examples:
            break
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source,target

class Vocab:
    def __init__(self, tokens, min_freq=1, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        counter = {}
        ## 如果tokens是二维列表，先扁平化
        if tokens and isinstance(tokens[0], list):
            tokens = [token for sentence in tokens for token in sentence]
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1

        self.token_to_idx = {}
        self.idx_to_token = []
        self.unk = 0

        self.token_to_idx['<unk>'] = 0
        self.idx_to_token.append('<unk>')
        for token in reserved_tokens:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(t) for t in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices] if 0 <= indices < len(self) else '<unk>'
        return [self.to_tokens(i) for i in indices]
def download_file(url, save_path):
    print(f"正在下载 {url} -> {save_path}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))

    with open(save_path, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=4096):
            f.write(chunk)
            downloaded += len(chunk)
            progress = (downloaded / total_size) * 100 if total_size else 0
            print(f"\r下载进度: {downloaded}/{total_size} ({progress:.1f}%)", end="")
    print("\n下载完成")
def truncate_pad(line,num_steps,padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps] ## 截断
    return line + [padding_token] * (num_steps - len(line)) ## 填充
def build_array_nmt(lines,vocab,num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines] ## lines转换成全部由数字组成的序列列表

    lines = [l + [vocab['<eos>']] for l in lines] # 给每一句末尾强制加上终止符 <eos>

    array = torch.tensor([truncate_pad(l,num_steps,vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"解压完成，目录：{extract_dir}")

def read_data_nmt():
    cache_dir = os.path.join(".", "data")
    zip_path = os.path.join(cache_dir, FILE_NAME)
    extract_dir = os.path.join(cache_dir, "fra-eng")

    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        download_file(FILE_URL, zip_path)

    if not os.path.exists(extract_dir) or not any(os.listdir(extract_dir)):
        extract_zip(zip_path, extract_dir)

    # 查找fra.txt文件
    txt_path = None
    for root, dirs, files in os.walk(extract_dir):
        if "fra.txt" in files:
            txt_path = os.path.join(root, "fra.txt")
            break

    if txt_path is None:
        raise FileNotFoundError("在解压目录中未找到 fra.txt 文件")

    print(f"找到文件：{txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()
def load_data_nmt(batch_size, num_steps, num_examples=600):
    # 1. 读取并清洗文本
    text = preprocess_nmt(read_data_nmt())
    # 2. 分词，截取指定样本数
    source, target = tokenize_nmt(text, num_examples)
    # 3. 构建词表，统一预留特殊符号（新增<unk>）
    reserved = ['<unk>', '<pad>', '<bos>', '<eos>']
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=reserved)
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=reserved)
    # 4. 转为固定长度张量 + 有效长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 5. 构造数据集 + DataLoader（替代d2l.load_array）
    dataset = TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,**kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,enc_X,dec_X,*args):
        enc_outputs = self.encoder(enc_X,*args)
        dec_state = self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_X,dec_state)
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带屏蔽的softmax交叉熵损失函数"""
    # pred的形状：（batch_size,num_steps,vocab_size)
    ## label的形状：(batch_size,num_steps)
    ## valid_len的形状：(batch_size)
    def forward(self,pred,label,valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights,valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss,self).forward(
            pred.permute(0,2,1),label
        )
        weighted_loss = (unweighted_loss * weights).mean(dim = 1)
        return weighted_loss

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def grad_clipping(model, theta):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_seq2seq(net,data_iter,lr,num_epochs,tgt_vocab,device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr = lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    loss_record = []
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2) ## 训练损失总和 ：总损失，总词元数
        for batch in data_iter:
            X,X_valid_len,Y,Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],device = device).reshape(-1,1)
            dec_input = torch.cat([bos,Y[:,:-1]],1)
            Y_hat,_= net(X,dec_input,X_valid_len) ##3 Y_hat：解码器输出预测得分  (batch, tgt_len, vocab_size)
            l = loss(Y_hat,Y,Y_valid_len)
            l.sum().backward() # 损失函数的标量进行“反向传播”
            grad_clipping(net,1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(),num_tokens)
        avg_loss = metric[0] / metric[1]
        loss_record.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            speed = metric[1] / timer.stop()
            print(f'epoch {epoch + 1:3d} | loss: {avg_loss:.3f} | {speed:.1f} tokens/sec | {device}')
    print(f"\n训练完成，最终平均loss: {loss_record[-1]:.3f}")
    return loss_record
## 训练
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs = 0.005,200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)

net = EncoderDecoder(encoder, decoder)
train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)



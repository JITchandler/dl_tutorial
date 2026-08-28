## 序列到序列学习 seq2seq
## seq2seq 是从一个句子生成另一个句子
## 编码器和解码器都是RNN
## 将编码器最后的隐藏状态来初始解码器来完成信息传递
## 常用BLEU来衡量生成序列的好坏
import collections
import math
import os
import time
import zipfile
import requests
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)
    def forward(self,X,*args):
        raise NotImplementedError
##解码器
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)
    ## 新增一个init——state函数，用于将编码器的输出，转化成编码后的状态
    def init_state(self,enc_outputs,*args):
        raise NotImplementedError

    ## 为了逐个生成长度可变的词元序列，解码器在每个时间步，都会将输入和编码后的状态映射成当前时间步的输出词元
    def forward(self,X,state):
        raise NotImplementedError

class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqEncoder,self).__init__(**kwargs)

        ## 嵌入层
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

    def forward(self,X,*args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1,0,2)
        # 如果未提及状态，则默认为0
        out,state = self.rnn(X)
        # output shape (num_steps, batch_size,num_hiddens)
        # state shape: (num_layers,batch_size,num_hiddens)
        return out,state
encoder = Seq2SeqEncoder(vocab_size = 10,embed_size = 8,num_hiddens = 16,num_layers = 2,dropout = 0)
encoder.eval()
X = torch.zeros((4,7),dtype=torch.long)
output,state = encoder(X)

class Seq2SeqDecoder(Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqDecoder,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens,num_hiddens,num_layers,dropout=dropout)
        ## 全连接层：隐状态映射为词汇得分
        self.dense = nn.Linear(num_hiddens,vocab_size)
    def init_state(self,enc_outputs,*args):
        ## enc_outputs = (encoder_output,encoder = state)
        ## 返回编码器最后时刻的多层隐藏状态，作为解码器初始state
        return enc_outputs[1]
    def forward(self,X,state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1,0,2)
        # 广播context，使其具有与X相同的num_steps

        # state[-1]：编码器最后一层的隐状态 (batch, num_hiddens)
        # repeat(时间步长,1,1) 复制context，每个时间步共享同一个全局上下文
        # context shape: (num_steps, batch, num_hiddens)
        context = state[-1].repeat(X.shape[0],1,1)

        # 3. 拼接词向量和上下文向量作为GRU输入
        X_and_context = torch.cat((X,context),dim = 2)
        # shape: (num_steps, batch, embed_size + num_hiddens)

        # 4. GRU前向传播，用上一步state循环生成
        output,state = self.rnn(X_and_context,state)

        # 5.全连接预测每个位置词汇，再换回batch优先格式
        output = self.dense(output).permute(1,0,2)
        # output: (batch_size, num_steps, vocab_size
        return output,state

decoder = Seq2SeqDecoder(vocab_size = 10,embed_size = 8,num_hiddens = 16,num_layers = 2,dropout = 0)
decoder.eval()
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
FILE_NAME = "fra-eng.zip"
FILE_URL = DATA_URL + FILE_NAME


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


def extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"解压完成，目录：{extract_dir}")

    # 列出解压后的文件结构
    print("解压后的文件结构：")
    for root, dirs, files in os.walk(extract_dir):
        level = root.replace(extract_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
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

def build_array_nmt(lines,vocab,num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines] ## lines转换成全部由数字组成的序列列表

    lines = [l + [vocab['<eos>']] for l in lines] # 给每一句末尾强制加上终止符 <eos>

    array = torch.tensor([truncate_pad(l,num_steps,vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def truncate_pad(line,num_steps,padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps] ## 截断
    return line + [padding_token] * (num_steps - len(line)) ## 填充

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

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def stop(self):
        return time.time() - self.start_time

def grad_clipping(model, theta):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
## loss function
def sequence_mask(X,valid_len,value = 0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen),dtype = torch.float32,device = X.device)[None,:] < valid_len[:,None]
    X[~mask] = value
    return X
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
## 训练
## 特定的序列开始词元（“<bos>”）和 原始的输出序列（不包括序列结束词元“<eos>”） 拼接在一起作为解码器的输入。 这被称为强制教学（teacher forcing）
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



## 9.6.3 合并编码器和解码器
## 我们可以将encoder 和 decoder合并，并且可以拥有可选的额外的参数
## 编码器的输出用于生成编码状态，这个状态又被解码器作为其输入的一部分

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,**kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,enc_X,dec_X,*args):
        enc_outputs = self.encoder(enc_X,*args)
        dec_state = self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_X,dec_state)
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr,num_epochs = 0.005,500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter,src_vocab,tgt_vocab = load_data_nmt(batch_size,num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

def predict_seq2seq(net,src_sentence,src_vocab,tgt_vocab,num_steps,device,save_attention_weights = False):
    """序列到序列模型的预测"""
    # 在预测的时候，将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)],device = device)
    src_tokens = truncate_pad(src_tokens,num_steps,src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens,dtype = torch.long,device = device),
        dim = 0
    )
    en_outputs = net.encoder(enc_X,enc_valid_len)
    dec_state = net.decoder.init_state(en_outputs,enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()

        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
##  预测序列的评估
def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')














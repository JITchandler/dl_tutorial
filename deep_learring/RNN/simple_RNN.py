## Simple_RNN_Version
##  we can use senior API to get the same model

## first ,we still load time machine

import math
import random
import time
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
batch_size, num_steps = 32, 35
import os
def load_time_machine_dataset():
    url = 'https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    cache_dir = os.path.join('data', 'timemachine')
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, 'timemachine.txt')

    if not os.path.exists(file_path):
        import urllib.request
        print(f'Downloading {url}...')
        urllib.request.urlretrieve(url, file_path)
        print('Download completed')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    text = ' '.join([line.strip() for line in lines])
    text = text.lower()
    text = text.replace(' ', '')
    return text
# 词汇表类
class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        counter = {}
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

# 随机采样迭代器
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_examples = (len(corpus) - 1) // num_steps
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_examples // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# 顺序分区迭代器（修复：指定 dtype=long）
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    # 顺序分区迭代器 标准实现
    offset = random.randint(0, num_steps)
    corpus = corpus[offset:]
    num_batches = (len(corpus) - 1) // (batch_size * num_steps)
    Xs = torch.tensor(corpus[: num_batches * batch_size], dtype=torch.long)
    Ys = torch.tensor(corpus[1: num_batches * batch_size + 1], dtype=torch.long)
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)
    num_steps_per_batch = Xs.shape[1] // num_steps
    for i in range(num_steps_per_batch):
        start = i * num_steps
        end = start + num_steps
        X = Xs[:, start:end]
        Y = Ys[:, start:end]
        yield X, Y

# 数据加载入口
def load_data_time_machine(batch_size, num_steps, use_random_iter=False):
    text = load_time_machine_dataset()
    tokens = list(text)
    vocab = Vocab(tokens)
    corpus = [vocab[t] for t in tokens]
    if use_random_iter:
        data_iter_fn = seq_data_iter_random
    else:
        data_iter_fn = seq_data_iter_sequential

    data_iter = data_iter_fn(corpus, batch_size, num_steps)
    return data_iter, vocab

train_iter, vocab = load_data_time_machine(batch_size, num_steps)

## define model
## senior API provide achievement of RNN,we create a RNN layer called rnn_layer
## with 256 hidden units in a hidden layer
num_hiddens = 256 #->
rnn_layer = nn.RNN(len(vocab), num_hiddens)

## we use tensor to init hidden state,and its shape is (hidden_layers,batch_size,hidden_units)
state = torch.zeros(1,batch_size, num_hiddens)
print(state.shape)

## we use hidden state and input to update output, and it should be emphasized that
## the output of rnn_layer do not involve output layer,it means state in each num_step
## we can use these state  as input for later outputs layer
X = torch.rand(size=(num_steps,batch_size,len(vocab)))
Y,state_new = rnn_layer(X, state)
print(Y.shape)
print(state_new.shape)

##  like 8.5,we need define RNNModel class,rnn_layer only include hidden layer
## we should create independent output layer
class RNNModel(nn.Module):
    """RNN model"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
    ## RNN是可以分为单向或者是双向，
    ## 单向RNN，隐藏层只存从前往后的提取的信息，当预测下一个字的时候，只用这一条25维向量，所以全连接层输入只需要256个通道
    ## 双向：同时读，正向就是从左到右，反向就是从右到左，每个字都会得到两端信息，正向提取的信息+反向提取的信息
    ## 模型会将两端向量拼到一起，合起来就是512维度，所以线性层输入维度就会翻倍,写出num_hidden * 2
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
    def forward(self,inputs,state):
            X = F.one_hot(inputs.T.long(), self.vocab_size)
            X = X.to(torch.float32)
            Y,state = self.rnn(X, state)
            # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
            # 它的输出形状是(时间步数*批量大小,词表大小)。
            output = self.linear(Y.reshape((-1, Y.shape[-1])))
            # Y.reshape(-1, hidden_dim)：把时间步、批量展平为二维
            # 变换前：(num_steps, batch, hidden_dim)
            # 变换后：(num_steps * batch, hidden_dim)
            # 线性层对每一个时序位置单独预测词表分布
            # output
            # 形状：(num_steps * batch, vocab_size)，每个行对应一个时刻的词分类得分
            return output,state
    def begin_state(self,device,batch_size = 1 ):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # LSTM 二元组 (h,c)
            h = torch.zeros(
                (
                    self.num_directions * self.rnn.num_layers,
                    batch_size,
                    self.num_hiddens,
                ),
                device=device
            )
            # 修复7：zeros_like复制h形状
            c = torch.zeros_like(h)
            return (h, c)

## 训练与预测
## 在训练模型之前，让我们基于一个具有随机权重的模型进行预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net  = RNNModel(rnn_layer,vocab_size= len(vocab))
net = net.to(device)
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    # 初始化隐状态：batch_size=1
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]

    get_input = lambda:torch.tensor([outputs[-1]],device=device).reshape((1,1))

    # 1. 预热阶段：把前缀逐个喂入网络，更新隐状态
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 2. 循环预测 num_preds 个字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # outputs.append(int(y.argmax(dim=1).reshape(1)))
        outputs.append(y.argmax(dim=1).item())
    # 索引转回字符并拼接
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 前缀去掉空格（数据集无空格），预测10个字符
pred_str = predict_ch8('timetraveller', 10, net, vocab, device)
print(pred_str)
# ## 解决梯度爆炸问题，进行梯度裁剪，限制所有参数的L2范数不超过阈值theta，梯度太大就等比例缩小
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
def train_epoch_ch8(net,train_iter,loss,updater,device,use_random_iter):
    ## use_random_iter:是否随机采用迭代器，随机采用每批无时序关联，顺序采用需要保留时序隐状态
    state = None
    start_time = time.time()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0 ## 总词元数量
    # use_random_iter = True：随机打乱取片段，前后批次文字毫无关联；
    # use_random_iter = False：顺序取文字，上一批的结尾就是下一批的开头，上下文连续。
    for X,Y in train_iter:
        batch_size = X.shape[0]
        if state is None or use_random_iter:
            ## 如果隐状态是空的或者采用随机存取，则需要重置隐状态
            state = net.begin_state(batch_size = X.shape[0], device=device)
        else :
            ## 若是顺序读取，则需要保留上一批记忆
            ## 把隐状态和之前所有批次的梯度链条间断，防止梯度爆炸
            ## 在细分两种模型，torch内置GRU，之间state.death()
            ##              torch内置LSTM / 你自己手写的RNNModelScratch：state 是元组包裹多个张量，要循环每一个张量分别切断梯度 for s in state: s.detach_()。
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                # 内置单层RNN，state为单张量
                 state.detach_()
            else:
                # 手写RNN / nn.LSTM/GRU，state是元组
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        ## 前向传播
        y_hat, state = net(X, state)
        l = loss(y_hat, y).mean()
        # 反向传播与参数更新
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(y.numel())

        # 手动累加，替代Accumulator.add()
        num_token = y.numel()
        total_loss += l * num_token
        total_tokens += num_token
        # 现在迭代器修复后不会触发，但保留兜底
    if total_tokens == 0:
        return float('inf'), 0.0
    avg_loss = (total_loss / total_tokens).detach().cpu().item()
    ppl = math.exp(avg_loss)
    total_time = time.time() - start_time
    token_speed = total_tokens / total_time
    return ppl, token_speed
# ====================== 训练启动代码（新增）======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)

# 初始随机权重预测
# 数据集已全部删除空格，前缀不要加空格
pred_str = predict_ch8('timetraveller', 10, net, vocab, device)
print("随机权重预测结果：", pred_str)

# 超参（修复学习率）
num_epochs = 500
lr = 0.2
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
use_random = False

# 用于绘图保存数据
epoch_list = []
ppl_list = []

# 训练循环（每轮重建迭代器，解决数据耗尽inf问题）
for epoch in range(num_epochs):
    train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=use_random)
    ppl, speed = train_epoch_ch8(net, train_iter, loss_func, optimizer, device, use_random)

    # 只保存有效困惑度（过滤inf）
    if ppl != float('inf'):
        epoch_list.append(epoch + 1)
        ppl_list.append(ppl)

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch + 1}, 困惑度 ppl: {ppl:.2f}, 每秒处理词元: {speed:.1f}")
        pred_text = predict_ch8('timetraveller', 30, net, vocab, device)
        print("生成文本：", pred_text, "\n")

# ==================== matplotlib 绘制困惑度下降曲线 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

plt.figure(figsize=(10, 5))
plt.plot(epoch_list, ppl_list, color='#2E86AB', linewidth=2, label='困惑度 Perplexity')
plt.xlabel("训练轮数 Epoch", fontsize=12)
plt.ylabel("困惑度 PPL", fontsize=12)
plt.title("RNN语言模型训练困惑度变化曲线", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()










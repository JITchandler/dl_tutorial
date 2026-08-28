## GRU 门控循环单元，用于解决传统RNN的梯度消失问题
## GRU其中有两大核心的门，分别是更新门和重置门
import math
import os
import random
import time

## 更新门（update gate）：主要是用来控制保留多少的历史记忆，和吸收多少的新的信息
## 值的范围在0 ~ 1之间，值越大，说明越要保留上一刻的状态，少用当前的输入

## 重置门 (reset gate) ,控制多少过去的信息
## 值的范围在0 ~ 1之间，值越小，越清空之前的状态，只关注当前的输入
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
batch_size , num_steps = 32,35
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

## 初始化模型参数

## 从标准差为 0.01 的高斯分布中提取权重，并将偏置项改为0，超参数num_hiddens 定义为隐藏单元的数量
def get_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(shape,device=device) * 0.01
    #生成指定形状标准正态分布随机数，×0.01 缩小权重，避免初始激活值过大饱和

    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device)
                )
    #. 复用工具 three ()：一次性生成一组 (W_x, W_h, bias)

    W_xz,W_hz,b_z = three() ## 更新门参数
    W_xr,W_hr,b_r = three() ## 重置门参数
    W_xh,W_hh,b_h = three() ## 候选隐状态参数

    ## 输出层参数
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device=device)

    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params :
        param.requires_grad_(True) ## 开启自动求导，用于反向传播更新
    return params

## 定义模型
## 首先我们还是定义隐状态的初始化函数 init_gru_state，函数返回一个形状为（批量大小，隐藏单元个数）的张量
## 张量的值全部为零

def init_gru_state(batch_size,num_hiddens,device):
    return (torch.zeros(batch_size,num_hiddens,device=device),)

## 现在定义门控循环单元模型，模型的架构基本上和传统的RNN单元是相同的，只是权重更新公式更为复杂
def gru (inputs,state,params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    outputs = []
    H, = state
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh)  + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs,dim = 0),(H,)


## 训练与预测
## 训练过程与之前的类似，训练过后，我们分别打印输出训练集的困惑度，以及前缀“time traveler”和“traveler”的预测序列上的困惑度。

vocab_size,num_hiddens,device = len(vocab),256,torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs,lr = 500,1
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.device = device
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)


    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
net = RNNModelScratch(vocab_size, num_hiddens, device,get_params,init_gru_state,gru)
# 损失函数
loss = nn.CrossEntropyLoss()
# 优化器：SGD作用于手写GRU全部参数
updater = torch.optim.SGD(net.params, lr=lr)
# 是否使用随机迭代器（False=顺序连续文本，训练效果更好）
use_random_iter = False

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
# 只在循环外加载一次数据集，避免重复IO
train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=use_random_iter)
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



import math
import random
import time
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import os

batch_size, num_steps = 32, 35
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# RNN 参数初始化
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h  = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q  = torch.zeros(num_outputs, device=device)
    ## 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化隐状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# RNN 前向逻辑
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:

        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, 0), (H,)

# RNN 封装类
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

# ---------------- 主运行逻辑（已修复重复Vocab） ----------------

    # 只调用一次，全局统一使用这一份 vocab
train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=False)
X, Y = next(iter(train_iter))
num_hiddens = 512
net = RNNModelScratch(
        len(vocab),
        num_hiddens,
        device,
        get_params,
        init_rnn_state,
        rnn
    )

state = net.begin_state(X.shape[0], device)
Y, net_state = net(X.to(device), state)
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
        outputs.append(int(y.argmax(dim=1).reshape(1)))
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

# ## 训练
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

# ---------------- 主程序入口 ----------------
if __name__ == "__main__":

    train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True)
    num_hiddens = 256
    net = RNNModelScratch(
        len(vocab),
        num_hiddens,
        device,
        get_params,
        init_rnn_state,
        rnn
    )

    # 超参数
    lr = 1
    num_epochs = 500
    loss_fn = nn.CrossEntropyLoss()

    # 自定义SGD更新函数
    def updater(num_tokens):
     with torch.no_grad():
        for p in net.params:
            if p.grad is not None:
                p.data -= lr * p.grad
                p.grad.zero_()
    # 绘图存储列表

    epoch_list = []
    ppl_list = []

    # 训练循环
    for epoch in range(num_epochs):
        train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=False)
        ppl, speed = train_epoch_ch8(net, train_iter, loss_fn, updater, device, use_random_iter=False)
        epoch_list.append(epoch + 1)
        ppl_list.append(ppl)
        if (epoch + 1) % 10 == 0:
            print(f"epoch:{epoch+1} 困惑度:{ppl:.2f} 训练速度:{speed:.0f} token/s")
            pred_text = predict_ch8('timetraveller', 80, net, vocab, device)
            print("生成文本：", pred_text, "\n")
    # 绘制困惑度曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_list, ppl_list, color="#1f77b4", linewidth=2, label="困惑度 ppl")
    plt.xlabel("训练轮数 epoch")
    plt.ylabel("困惑度 ppl")
    plt.title("RNN训练困惑度变化曲线")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


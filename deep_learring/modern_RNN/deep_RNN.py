import math
import os
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

# 超参
batch_size, num_steps = 32, 35

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

# 顺序分区迭代器
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    corpus = corpus[offset:]
    num_batches = (len(corpus) - 1) // (batch_size * num_steps)
    Xs = torch.tensor(corpus[: num_batches * batch_size * num_steps], dtype=torch.long)
    Ys = torch.tensor(corpus[1: num_batches * batch_size * num_steps + 1], dtype=torch.long)
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
    data_iter_fn = seq_data_iter_random if use_random_iter else seq_data_iter_sequential
    return data_iter_fn(corpus, batch_size, num_steps), vocab

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size).to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            h = torch.zeros((self.num_directions * self.rnn.num_layers,
                             batch_size, self.num_hiddens), device=device)
            c = torch.zeros_like(h)
            return (h, c)

# 梯度裁剪
def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# 文本预测（加no_grad、eval模式）
def predict_ch8(prefix, num_preds, net, vocab, device):
    net.eval()
    with torch.no_grad():
        state = net.begin_state(batch_size=1, device=device)
        outputs = [vocab[prefix[0]]]
        get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

        for y in prefix[1:]:
            _, state = net(get_input(), state)
            outputs.append(vocab[y])

        for _ in range(num_preds):
            y, state = net(get_input(), state)
            idx = int(y.argmax(dim=1).cpu())
            outputs.append(idx)
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 单轮训练
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter, clip_theta=1):
    state = None
    start_time = time.time()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0

    for X, Y in train_iter:
        batch_sz = X.shape[0]
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=batch_sz, device=device)
        else:
            if isinstance(state, tuple):
                for s in state:
                    s.detach_()
            else:
                state.detach_()

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y).mean()

        updater.zero_grad()
        l.backward()
        grad_clipping(net, clip_theta)
        updater.step()

        num_token = y.numel()
        total_loss += l * num_token
        total_tokens += num_token

    if total_tokens == 0:
        return float('inf'), 0.0
    avg_loss = (total_loss / total_tokens).detach().cpu().item()
    # 防止exp溢出
    ppl = math.exp(min(avg_loss, 100))
    total_time = time.time() - start_time
    token_speed = total_tokens / total_time
    return ppl, token_speed

# ===================== 主训练入口 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_layers = 2
num_hiddens = 256
use_random_iter = False

# 仅加载一次数据集，优化速度
train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter)
vocab_size = len(vocab)

# 深层LSTM（num_layers=2，对应深层循环神经网络）
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens, num_layers=num_layers,dropout=0.2)
net = RNNModel(lstm_layer, vocab_size).to(device)

num_epochs, lr = 500, 0.5
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

epoch_list = []
ppl_list = []

for epoch in range(num_epochs):
    # 每轮重新生成迭代器，不重复加载文本
    train_iter, _ = load_data_time_machine(batch_size, num_steps, use_random_iter)
    net.train()
    ppl, speed = train_epoch_ch8(net, train_iter, loss_func, optimizer, device, use_random_iter)

    if ppl != float('inf'):
        epoch_list.append(epoch + 1)
        ppl_list.append(ppl)

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch + 1}, 困惑度 ppl: {ppl:.2f}, 每秒词元: {speed:.1f}")
        pred_text = predict_ch8('timetraveller', 30, net, vocab, device)
        print("生成文本：", pred_text, "\n")

# 绘图兼容多系统字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 5))
plt.plot(epoch_list, ppl_list, color='#2E86AB', linewidth=2, label='困惑度 Perplexity')
plt.xlabel("训练轮数 Epoch", fontsize=12)
plt.ylabel("困惑度 PPL", fontsize=12)
plt.title("深层LSTM语言模型训练困惑度变化曲线", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()
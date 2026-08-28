import collections
import re
import urllib.request
import matplotlib.pyplot as plt
import ssl
# 关闭SSL证书校验，解决ASN1报错
ssl._create_default_https_context = ssl._create_unverified_context

# 下载地址
url = "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"

# 读取并清洗文本
def read_time_machine():
    with urllib.request.urlopen(url) as f:
        lines = f.read().decode('utf-8').splitlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

# 词元化
def tokenize(lines, token='word'):
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)

# 统计词频
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 词表类
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._tokens_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._tokens_freq:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._tokens_freq
corpus = [token for line in tokens for token in line]
vocab = Vocab(tokens)
print("出现频率最高的前10个词：")
print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
print("词频列表前20项：", freqs[:20])
freqs = [freq for token, freq in vocab.token_freqs]

# 绘制双对数词频图（正常弹窗）
plt.figure(figsize=(8, 6))
plt.plot(freqs)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('token: x')
plt.ylabel('frequency: n(x)')
plt.title('Token Frequency (Log-Log Scale)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]

# 3. 构建二元词表（复用原生 Vocab，完全等价 d2l.Vocab）
bigram_vocab = Vocab(bigram_tokens)
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]

# 4. 输出出现频率最高的前10个二元词组
print("Top 10 Bigram 词对及频次：")
print(bigram_vocab.token_freqs[:10])


# ========== 三元语法 Trigram（你要的代码） ==========
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)  # 替换 d2l.Vocab
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

print("\nTop10 三元词组：")
print(trigram_vocab.token_freqs[:10])

 ##========= 原生 Matplotlib 绘图（替代 d2l.plot） =========
plt.figure(figsize=(9, 6))
# 绘制三条曲线
plt.plot(freqs, label='unigram')
plt.plot(bigram_freqs, label='bigram')
plt.plot(trigram_freqs, label='trigram')

# 对数坐标轴、标签、图例、网格
plt.xscale('log')
plt.yscale('log')
plt.xlabel('token: x')
plt.ylabel('frequency: n(x)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.title('Unigram / Bigram / Trigram Frequency (Log-Log)')
plt.show()

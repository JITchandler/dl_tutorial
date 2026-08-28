## 文本预处理

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import collections
import re
import urllib.request

# 直接从官方地址下载 timemachine.txt
url = "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"


# 读取并清洗文本
def read_time_machine():
    # 下载文本
    with urllib.request.urlopen(url) as f:
        lines = f.read().decode('utf-8').splitlines()

    # 清洗：只保留字母，转小写，去空格
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# 运行
lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
## 词元
def tokenize(lines, token='word'):  # 标准：word 不是 words
    """将文本行拆分为单词或字符词元"""
    if token == "word":             # 缩进对齐！
        return [line.split() for line in lines]  # 按空格切分单词
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
## 调用词元化
tokens = tokenize(lines)
# for i in range(10):
#     print(tokens[i])

## 词表
## 词元是字符串类型，而模型需要的是数字，所以我们现在要构建词表，将字符串类型的词元映射到数字索引中，同时也支持反向索引，
## 先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结果称之为语料（corpus）。
## 然后根据每个唯一词元的出现频率，为其分配一个数字索引
## 另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。 我们可以选择增加一个列表，用于保存那些被保留的词元， 例如：填充词元（“<pad>”）； 序列开始词元（“<bos>”）； 序列结束词元（“<eos>”）。

def count_corpus(tokens):
    """统计词元的频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens = None,min_freq = 0,reserved_tokens = None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        ## 1.统计词频，按频率降序排序
        counter = count_corpus(tokens)
        self._tokens_freq = sorted(counter.items(),key=lambda x:x[1],reverse=True)
        ## 2.初始化特殊标记
        ## 索引 -》 词
        self.idx_to_token = ['<unk>'] + reserved_tokens
        ## 用枚举 enumerate 生成反向字典：
        self.token_to_idx = {token:idx for idx,token in enumerate(reserved_tokens)}

        ## 3.把“高频词”加入词表
        for token,freq in self._tokens_freq:
            if freq < min_freq: ## 频率过小，直接舍弃
                break
            if token not in self.token_to_idx: ## 防止特殊标记重复
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self,tokens):
        """词元转索引，支持单个词元或者列表"""
        if not isinstance(tokens, (list,tuple)):
             return self.token_to_idx.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self,indices):
        """索引转词元，支持单个索引或列表"""
        if not isinstance(indices, (list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """未知词元的索引"""
        return 0
    @property
    def token_freqs(self):
        """词频列表"""
        return self._tokens_freq

vocab = Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

## 整合所有功能
## 我们将所有功能打包到load_corpus_time_machine函数中， 该函数返回corpus（词元索引列表）和vocab（时光机器语料库的词表）
## 我们要做出一些改变
## 为了简化后面章节的训练，我们采用字符，而不是单词 实现文本词元化
## 返回的corpus仅处理单个列表，而不是使用
def load_corpus_time_machine(max_tokens = -1):
    """返回时光机器数据集的词元素索引表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines,'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus,vocab
corpus,vocab = load_corpus_time_machine()
print(len(corpus))
print(len(vocab))













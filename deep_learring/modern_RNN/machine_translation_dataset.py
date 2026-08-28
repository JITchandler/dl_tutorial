## 9.5 机器翻译与数据集
## 机器翻译是将输入序列装换成输出序列的序列转换模型的核心问题
## 机器翻译是将序列从一种语言自动翻译成另外一种语言
##其中一直使用统计学方法来进行翻译称为统计机器翻译，采用基于神经网络方法的翻译被称为神经机器翻译
## 神经网络机器翻译强调端到端的学习，与之前的单一语言的语言模型存在不同
## 机器翻译的数据集是有源语言和目标语言的 文本序列对 组成 所以要采用全新的方法，而不是复用语言模型的预处理程序

import os
import torch
import zipfile
import requests
import hashlib
import matplotlib.pyplot as plt
import numpy as np

from torch.testing._internal.distributed import rpc
from torch.utils.data import TensorDataset, DataLoader

## 9.5.1 下载和预处理数据集
## 下载一个由Tatoeba项目的双语句子对 组成的“英－法”数据集，数据集中的每一行都是制表符分隔的文本序列对， 序列对由英文文本序列和翻译后的法语文本序列组成。

## 数据集地址和校验码
# 数据集地址
# 数据集地址
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

raw_text = read_data_nmt()
print(raw_text[:75])

## 原始文本数据需要经过几个处理
## 用空格代替不间断空格
## 使用小写字幕替换大写字母，并在单词和标点符号之间插入空格

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
text = preprocess_nmt(raw_text)
print(text[:80])

## 词元化
## 使用tokenize_nmt对num_examples个文本序列进行词元化，每个词元要么是一个词，要么是一个标点符号
## 函数返回两个列表，source,target 分别表示英语和法语

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
source,target = tokenize_nmt(text)
print(source[:6])
print(target[:6])


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制两组序列长度分布直方图，无d2l依赖"""
    # 设置画布大小，等价 d2l.set_figsize()
    plt.figure(figsize=(6, 4))

    # 统计每个句子的token长度
    len_x = [len(seq) for seq in xlist]
    len_y = [len(seq) for seq in ylist]
    data = [len_x, len_y]

    # 绘制双组直方图
    n, bins, patches = plt.hist(data, bins=30, label=legend)

    # 给第二组填充斜纹 /，和原代码效果一致
    for patch in patches[1]:
        patch.set_hatch('/')

    # 坐标轴与图例
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


# # 调用方式不变
# show_list_len_pair_hist(
#     ['source', 'target'],
#     '# tokens per sequence',
#     'count',
#     source,
#     target
# )

## 9.5.3 词表
## 我们需要构建源语言和目标语言两个词表，使用单词级词元化，词表大小将明显大于字符级次元话时候的词表大小
## 为了缓解，我们将出现次数少于2次的低频词元是为相同的未知（“<unk>”）词元。
## 除此之外，我们还指定了额外的特定词元， 例如在小批量时用于将序列填充到相同长度的填充词元（“<pad>”）， 以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）。
class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
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

reserved = ['<pad>', '<bos>', '<eos>']
## 自动补上未知词 <unk>，放到特殊标记最后
if '<unk>' not in reserved:
    reserved.insert(0,'<unk>')
src_vocab =Vocab(source,min_freq=2,reserved_tokens=reserved)
print(len(src_vocab))

## 加载数据集
## 在机器翻译中，每个样本都是由源和目标组成的文本序列对， 其中的每个文本序列可能具有不同的长度。
## 假设同一个小批量中的每个序列都应该具有相同的长度num_steps， 那么如果文本序列的词元数目少于num_steps时， 我们将继续在其末尾添加特定的“<pad>”词元， 直到其长度达到num_steps； 反之，我们将截断文本序列时，只取其前num_steps 个词元， 并且丢弃剩余的词元。

def truncate_pad(line,num_steps,padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps] ## 截断
    return line + [padding_token] * (num_steps - len(line)) ## 填充

print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))


def build_array_nmt(lines,vocab,num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines] ## lines转换成全部由数字组成的序列列表

    lines = [l + [vocab['<eos>']] for l in lines] # 给每一句末尾强制加上终止符 <eos>

    array = torch.tensor([truncate_pad(l,num_steps,vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len
## 训练模型
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

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break




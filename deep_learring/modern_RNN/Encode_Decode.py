## 类似机器翻译的那种，序列模型转换的问题，其要求就是通过输入大小可变的序列，转换成输出大小可变的序列
## 我们提出了编码器 - 解码器的概念
## 编码器（encoder）：它接受一个长度可变的序列作为输入， 并将其转换为具有固定形状的编码状态。
## 解码器（decoder）: 它将固定形状的编码状态映射到长度可变的序列

## 过程：首先编码器将一个输入序列编码成一个状态，之后解码器将状态进行解码，一个词元，一个词元的生成翻译后的序列进行输出

## 编码器：
import torch
from torch import nn
## 在编码器接口中，我们指定可变序列X作为编码器的输入X，任何继承这个Encoder基类的模型将完成代码实现
class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)
    def forward(self,x,*args):
        raise NotImplementedError

##解码器
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)
    ## 新增一个init——state函数，用于将编码器的输出，转化成编码后的状态
    def int_state(self,enc_outputs,*args):
        raise NotImplementedError

    ## 为了逐个生成长度可变的词元序列，解码器在每个时间步，都会将输入和编码后的状态映射成当前时间步的输出词元
    def forward(self,X,state):
        raise NotImplementedError

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

## 小结：
## “编码器－解码器”架构可以将长度可变的序列作为输入和输出，因此适用于机器翻译等序列转换问题。
##  编码器将长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。
##  解码器将具有固定形状的编码状态映射为长度可变的序列。


## 背景: 传统的Seq2Seq翻译模型，编码器将整段输入压缩成唯一固定向量，长句子的时候容易信息丢失

## Bahadnau 软注意力，解码器在每一步翻译的时候，动态看输入句子的不同位置，分配不同的权重，不再只用一个全局向量

## 流程分为三步
## 1.计算得分，当前解码器的隐状态S和编码器每时刻隐状态H做匹配
## 2.softmax归一化得到注意力权重
## 3.加权求和得到上下文向量context
## 4.把ct和当前解码器状态拼接，预测下一个单词
import torch
from torch import nn

from deep_learring.Attention_Mechanism.attention_prompt import attention_weights
from deep_learring.Attention_Mechanism.attention_scoring import masked_softmax, show_heatmaps
from deep_learring.modern_RNN.Encode_Decode import EncoderDecoder
from deep_learring.modern_RNN.seq2seq import Seq2SeqEncoder, load_data_nmt, train_seq2seq, predict_seq2seq, bleu


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
class AdditiveAttention(nn.Module):
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super(AdditiveAttention,self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size,num_hiddens,bias = False)
        self.W_q = nn.Linear(query_size,num_hiddens,bias = False)
        self.W_v = nn.Linear(num_hiddens,1,bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values,valid_lens):
        queries , keys = self.W_q(queries),self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        # queries.unsqueeze(2)：新增第 2 维 → (batch, 查询数, 1, h)
        # keys.unsqueeze(1)：新增第 1 维 → (batch, 1, 键值数, h)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # self.W_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores,valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights),values)
## 带有注意力机制解码器
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self,**kwargs):
        super(AttentionDecoder,self).__init__(**kwargs)
    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqAttentionDecoder,self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens,num_hiddens,num_hiddens,dropout)
        ## 词嵌入层，单词索引 -> 稠密向量
        self.embedding =nn.Embedding(vocab_size,embed_size)
        ## GRU循环单元，输入维度 = 词嵌入 + 注意力上下文
        self.rnn = nn.GRU(embed_size + num_hiddens,num_hiddens,num_layers = num_layers,dropout = dropout)
        ## 输出全连接：隐状态 -> 词汇表概率分布
        self.dense = nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,enc_valid_lens,*args):
        # outputs的形状为（batch_size,num_steps,num_hiddens）
        # hidden_state的形状为（num_layers,batch_size,num_hiddens）
        outputs,hidden_state = enc_outputs
        # 交换维度：(seq_len, batch, hidden)，方便后续循环遍历
        return (outputs.permute(1,0,2),hidden_state,enc_valid_lens)
    def forward(self,X,state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,num_hiddens)
        enc_outputs,hidden_state,enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1,0,2)
        outputs,self._attention_weights = [],[]
        for x in X:
            # hidden_state[-1]：解码器最后一层GRU隐状态 s_t
            # query形状：(batch, 1, hidden) 单步查询向量
            query = torch.unsqueeze(hidden_state[-1],1)

            ## 注意力计算 ： Bahdanau 加性注意力
            ## query=解码器状态，key/value = 编码器的全部输出
            # context: (batch, 1, hidden) 加权求和后的上下文向量 c_t
            context = self.attention(query,enc_outputs,enc_outputs,enc_valid_lens)

            x = torch.cat((context,torch.unsqueeze(x,dim =1,)),dim = -1)
            ## 送入GRN更新解码器隐状态
            # x转置为(1, batch, embed+hidden) 符合GRU输入格式
            out,hidden_state = self.rnn(x.permute(1,0,2),hidden_state)
            # 保存当前步输出、注意力权重（用于可视化对齐）
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 拼接所有时间步输出，过全连接预测单词
        outputs = self.dense(torch.cat(outputs,dim = 0))
        ## # 换回 (batch, tgt_len, vocab_size) 标准输出格式
        return outputs.permute(1,0,2),[enc_outputs,hidden_state,enc_valid_lens]
    @property ## 取出注意力权重
    def attention_weights(self):
        return self.attention_weights

encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()

decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
# print(output.shape)
# print(len(state))
# print(state[0].shape)
# print(len(state[1]))
# print(state[1][0].shape)

## 训练
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs = 0.005, 250
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter,src_vocab,tgt_vocab = load_data_nmt(batch_size,num_steps)
encoder = Seq2SeqEncoder(len(src_vocab),embed_size,num_hiddens,num_layers,dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder,decoder)
train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {bleu(translation, fra, k=2):.3f}')


# show_heatmaps(
#     attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
#     xlabel='Key positions', ylabel='Query positions')






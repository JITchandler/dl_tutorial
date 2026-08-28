## attention_prompt 注意力提示

## 核心概念：不会平等看到所有的信息，只会关注重要的部分，忽略无关的内容
## 基本流程：将输入映射为三组向量 Query(Q),Key(K),Value(V)
## Q:为想要生成的内容，K: 输入原文特征 ,V：需要提取的信息
## 计算Q和K的相似度，得到注意力分数，之后采用softmax分数，得到权重，用权重加权所有V，输出聚焦关键信息的向量
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F, attention


def show_heatmaps(matrices,xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    ## 获取子图行列的布局
    num_rows,num_cols = matrices.shape[0], matrices.shape[1]
    ## 创建画布与子图网络
    fig , axes = plt.subplots(num_rows , num_cols, figsize=(figsize[0]*num_cols, figsize[1]*num_rows),
                             sharex=True, sharey=True, squeeze=False)
    pcm = None # 保存最后一张图用于色条
    for i,(row_axes,row_matrices) in enumerate(zip(axes,matrices)):
         for j,(ax,matrix) in enumerate(zip(row_axes,row_matrices)):
             # tensor转 numpy,脱离计算图
            data = matrix.detach().cpu().numpy()
            pcm = ax.imshow(data, cmap=cmap)
             # 仅最后一行显示X标签
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            # 仅第一列显示y标签
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles is not None:
                ax.set_title(titles[j])
    # 全局颜色条：
    fig.colorbar(pcm,ax=axes,shrink=0.6)
    # 展示图像
    plt.show()
# attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
# show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
# X = torch.rand(5,5)
# X = F.softmax(X,dim=1)
# X = X.reshape((1,1,5,5))
# show_heatmaps(X, xlabel='Keys', ylabel='Queries')


## 注意力汇聚
## 生成数据集，50个训练样本和50个测试样本
n_train = 50 # 训练样本数量
x_train,_ = torch.sort(torch.rand(n_train) *5) ## 排序后的训练样本数量

def f(x):
    return 2 * torch.sin(x) + x**0.8
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
print(n_test)

def plot_kernel_reg(y_hat):
    # 绘制真实曲线，预测曲线
    plt.plot(x_test,y_truth,label='Truth')
    plt.plot(x_test,y_hat,label='Pred')

    # 绘制训练离散点
    plt.scatter(x_train,y_train,marker ='o',alpha = 0.5)
    # 坐标轴、图例、范围设置
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.xlim(0, 5)
    plt.ylim(-1, 5)
    plt.show()
## 平均汇聚
## 通过计算输入特征图每个子区域的平均值来提取特征
# y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# plot_kernel_reg(y_hat)

## 非参数注意力汇聚
# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))

# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
attention_weights= nn.functional.softmax(-(X_repeat - x_train)**2 / 2,dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = torch.matmul(attention_weights,y_train)
plot_kernel_reg(y_hat)
show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')


## 带有参数注意力汇聚

## 非参数的Nadaraya-Wastson核回归具有一致性的优点：如果有足够多的数据，模型可以收敛到最优结果
## 尽管如此，我们还是可以可学习的参数集成到注意力汇聚中，
X = torch.ones((2,1,4))
Y = torch.ones((2,4,6))
print(torch.bmm(X,Y).shape)

## 在注意力机制中，我们可以使用小批量矩阵乘法来计算小批量数据中的加权平均值
## 这里就是注意力加权求和，权重全部均等 0.1，相当于对每行values求均值
weights = torch.ones((2,10)) * 0.1
values  = torch.arange(20.0).reshape((2,10))

# print(torch.bmm(weights.unsqueeze(1),values.unsqueeze(-1)))

## 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,),requires_grad=True))
    def forward(self,queries,keys,values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        ## 复制扩展查询，使得和 key 匹配
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1,keys.shape[1]))

        ##计算带可学习w的注意力权重
        self.attention_weights =F.softmax(-((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                     values.unsqueeze(-1)).reshape(-1)

## 训练

# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss  = nn.MSELoss(reduction='none')
trainer  =torch.optim.SGD(net.parameters(),lr=0.5)
# 存储历史数据，替代Animator
epoch_list = []
loss_list = []

# 初始化画布
plt.figure(figsize=(6,4))

for epoch in range(10):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    total_loss = l.sum()
    total_loss.backward()
    trainer.step()

    # 保存数据
    epoch_list.append(epoch + 1)
    loss_list.append(total_loss.item())
    print(f'epoch {epoch + 1}, loss {float(total_loss.detach()):.6f}')

    # 清空画布，重新绘制loss曲线
    plt.cla()
    plt.plot(epoch_list, loss_list, marker='o', c='#1f77b4')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(1, 5)
    plt.grid(alpha=0.3)
    plt.title('Training Loss Curve')
    plt.pause(0.5)  # 停留0.5秒，动态动画效果

# 循环结束后保留最终图像
plt.show()

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')







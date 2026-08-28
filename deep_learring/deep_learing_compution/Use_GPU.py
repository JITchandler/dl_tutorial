import torch
import numpy as np
from torch import  nn
# print(torch.device('cpu'))
# print(torch.device('cuda'))
# print(torch.device('cuda:1'))
print(torch.cuda.device_count())
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
# print(try_gpu())
# print(try_gpu(10))
# print(try_all_gpus())
## 张量和GPU
## 我们可以查询张量所在的设备，默认情况下，张量是在CPU上面创建的
x =torch.tensor([1,2,3])
print(x.device)
## 注意的是，当我们需要对张量进行操作的时候，必须保证张量都在同一个设备上，否则框架不知在哪里存储结果，甚至不知道在哪里进行计算

##存储在GPU上
## 我们可以在创建张量的时候，指定张量在GPU上面创建，此时张量消耗显存，但是要确保不超过最大的显存
y = torch.ones(2,3,device = try_gpu())
print(y)

Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)
## 当拥有两张GPU时，此刻如果需要操作的张量不在同一个设备中的时候，需要进行张量复制，之后进操作
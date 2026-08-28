# 跃迁函数的表示

import numpy as np
def step_function(x):
    return np.array( x > 0 , dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
# 定义ReLu函数
def ReLU(x):
    return np.maximum(0, x)
# 定义softmax函数
def softmax0(x):
    return np.exp(x)/ np.sum(np.exp(x))

#定义 参数为矩阵的 softmax参数
def softmax(x):
    #如果是二维矩阵
    if x.ndim == 1:
     x = x.T
     x = x - np.max(x ,axis = 0)
     y = np.exp(x)/np.sum(np.exp(x),axis = 0)
     return y.T
    # 防溢出处理
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))
def identity(x):
    return x
if __name__ == "__main__":
        x = np.array([0 ,1 ,2 ,3, 4, 5 ,-1,-2,-3,-4,-5])
        print(step_function(x))
        print(sigmoid(x))
        print(tanh(x))
        print(ReLU(x))
        print(softmax0(x))
        X = np.array([[0,1,2],[3,4,5],[7,8,9],[10,11,12]])
        print(softmax(X))
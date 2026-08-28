import numpy as np
from numpy.ma.core import identity

from common.functions import sigmoid,identity


def init_network():
    network = {}
    #定义第一层
    network['w1']  = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']  = np.array([0.1,0.2,0.3])

    # 定义第二层
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    # 定义第三层
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):

    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    b1,b2 ,b3= network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3) + b3
    y = identity(a3)
    return y

network = init_network()
x = np.array([1,2])
y = forward(network, x)
print(y)




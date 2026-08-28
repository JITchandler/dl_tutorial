import numpy as np
#生成随机浮点数
arr = np.random.random((2,3))
print(arr)

# 生成随机整数， 范围在0~5
arr1 = np.random.randint(0,5,(2,3))
print(arr1)
#生成随机浮点数，范围在0~2
arr2 = np.random.uniform(0,2,(2,3))
print(arr2)
# 随机种子，可以让随机生成的矩阵，每次生成的结果都一样
np.random.seed(10)
arr3 = np.random.randint(0,2,(2,3))
print(arr3)
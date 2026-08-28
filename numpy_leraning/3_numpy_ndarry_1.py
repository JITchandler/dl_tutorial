import numpy as np
#ndarry的创建
arr = np.array([1,2,3,4,5])
print(arr)

#copy
arr1 = np.copy(arr) # 深拷贝，与原来的数组不一样
print('copy的数组',arr1)
# 预定义
# 全0 全1  未初始化 固定值

#全 0
arr2 = np.zeros((2,3)) #打印2行3列的数组，而且元素全为 0
print(arr2)
# 全 1
arr3 = np.ones((2,3)) #打印2行3列的数组，而且元素全为 1
print(arr3)

# 未初始化

arr4 = np.empty((4,3)) # 未初始化，矩阵内的内容是随机的
print(arr4)

#固定值
arr5 = np.full((2,3), 5) #固定值，元素全为5的2行3列的矩阵
print(arr5)
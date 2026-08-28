
import numpy as np
arr = np.array(5) #创建0维的array数组 单个数字
print(arr)
print("array的维度是",arr.ndim) #打印array的维度


arr2 = np.array([1,2,3])#创建1维的array数组 一维数组，中间用[] 包围
print(arr2)
print("arr2的维度是",arr2.ndim) #打印array2的维度


#同质性
arr3 = np.array([1,"hello"])
print("同质性：会将不同类型的数据强制转换成相同类型的",arr3)
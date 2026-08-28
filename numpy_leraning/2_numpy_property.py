#ndarray的属性
import numpy as np
array = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("数组的形状",array.shape) #shape属性，打印数组的行，列数
print("数组的维度",array.ndim)  # ndim 数组的维度
print("数组内元素的个数",array.size)  # 元素的个数
print("数组转置", array.T)  # 数组转置
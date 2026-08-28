# 认识numpy的常用函数
# sqrt(),exp()
import numpy as np
arr1 = np.sqrt(9)
print(arr1) #返回浮点数

#指数
arr2 = np.exp(1)
print(arr2)

#求对数
arr3 = np.log(np.exp(1))
print(arr3)

#绝对值
arr4 = np.array([-11,-2,3])
print(np.abs(arr4))

#四舍五入  rand() 四舍五入函数
arr5 = np.array([2.5,6.7,1.3])
print(np.round(arr5))
#向上取整 ceil() 天花板
print(np.ceil(arr5))
#向下取整 floor() 地板
print(np.floor(arr5))

#检测缺失值 Nan
print(np.isnan([1, 2, np.nan, 3]))


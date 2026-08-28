from xml.sax.handler import property_dom_node

import pandas as pd


#Series的属性

s = pd.Series([1,2,3,4,5,6],index=['c','d','e','f','g','h'])
s.name = "test"
print(s.index) # index属性
print(s.values)
print(s.name)
print(s.shape,s.size,s.ndim)
print(s.loc['c']) #显示索引，只能填写一个,填写的索引是你自己定义的 或者按照切片
print(s.iloc[1]) # 隐式索引，系统默认是从0开始的索引 或者按照切片
print(s.at['c']) #使用标签来访问单个元素
print(s.iat[1]) # 使用位置来访问单个元素

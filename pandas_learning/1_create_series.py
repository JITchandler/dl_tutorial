import pandas as pd
#创建series
#通过列表来创建series
s = pd.Series([1,2,3,4,5,6])
print(s)

#自定义series的索引
s = pd.Series([1,2,],index=['c','b'])
print(s)
#通过索引来寻找元素
s1 = pd.Series(s,index=['c'])
print(s1)

#通过字典来创建series
s2 = pd.Series({'a':1,'b':2,'c':3},index=['a','b'])
print(s2)
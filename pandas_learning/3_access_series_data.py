import pandas as pd
# s = pd.Series([1,2,3,4,5,6],index=['c','d','e','f','g','h'])
# #访问series数据
# print(s['f'])
# print(s[s>3])
# print(s.head()) #返回前五个
# print(s.tail()) #返回后五个

# series 的常用方法
s = pd.Series([1,2,None,4,5,6],index=['c','d','e','f','g','h'])
print(s)
print(s.describe())
print(s.count()) #忽视缺失值、
print(s.isin([2]))  #判断 数据 2 是否在Series中
print(s.isna())    #判断是否有缺失值
print(s.sum()) #总和
print(s.mean) #平均值
print(s.median()) #中位数
print(s.min()) #最小值
print(s.max()) #最大值
print(s.nunique()) #
print(s.var()) #方差
print(s.std()) #标准差

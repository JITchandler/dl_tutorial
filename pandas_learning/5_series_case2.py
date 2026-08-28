import numpy as np
import pandas as pd
# 给定某城市一周每天的最高温度Series，
# 完成以下任务:找出温度超过30度的天数
# 计算平均温度
# 将温度从高到低排序
# 找出温度变化最大的两天
temperatures = pd.Series([28, 31, 29, 32, 30, 27, 33],index=['周一','周二','周三','周四','周五','周六','周日'])
print(temperatures)
print('温度超过30度的天数',temperatures[temperatures>30].count())
print('平均温度:',temperatures.mean())
temperatures.sort_values(ascending=False)
print(temperatures.sort_values(ascending=False)) #降序
t = temperatures.diff().abs()
print('温度变化最大的两天',t.sort_values(ascending=False).keys()[0:2]) #每一天和前一天比较的变化情况,采用切片的方式



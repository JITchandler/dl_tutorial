import pandas as pd
#某产品过去12个月的销售量Series:
# 计算季度平均销量(每3个月为一个季度)
# 找出销量最高的月份
# 计算月环比增长率
#找出连续增长超过2个月的月份
Sales = pd.Series([120, 135, 145,160,155, 170, 180,175,190, 200, 210,220],index=pd.date_range('2022-01-01', periods=12, freq='MS'))
print(Sales)
qs = Sales.resample('QS').mean()
print(qs)
print(Sales.max())
t = Sales.pct_change() *100
print(t.round(2))

g = Sales.diff()


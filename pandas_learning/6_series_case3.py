import numpy as np
import pandas as pd
#给定某股票连续10个交易日的收盘价Series:
# 计算每日收益率(当日收盘价/前日收盘价-1)
# 找出收益率最高和最低的日期计算波动率(收益率的标准差)
prices = pd.Series([102.3,103.5,105.1,104.8,106.2,107.0,106.5,108.1,109.3, 110.2], index=pd.date_range('2023-01-01', periods=10))
print(prices)
#计算每天的收益率
a = prices.pct_change()
print(a.idxmax()) #收益率最高
print(a.idxmin()) #收益率最低
print(a.std()) # 波动率



from cProfile import label

import pandas as pd
import numpy as np
# 导入库
# 数据导入
# 数据清晰
# 缺失值的检查
# 数据特征的构造
# 数据分析

df = pd.read_csv('../data/penguins.csv')
# 原先一共有343个数据，经过数据清晰之后，现在有333个有效的数据
df.dropna(inplace=True) #删除 DataFrame 里所有包含缺失值（NaN / 空值）的整行数据，并且直接修改原数据，不生成新表格。
df['sex'] = df['sex'].astype('category')
# print(df.head())
df['bill_radio'] = df['bill_length_mm'] / df['bill_depth_mm']
# print(df.head())
labels = ['低','中','高']
df['mass_level'] = pd.cut(df['body_mass_g'],bins = 3, labels = labels)
print(df.tail())
print(df['mass_level'].value_counts())

result = df.groupby(['sex','island']).agg(
    {
        'body_mass_g':['mean','count']
    }
)
print(result)






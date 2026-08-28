from operator import index

import pandas as pd

s1 = pd.Series([1,2,3,4,5,6])
s2 = pd.Series([7,8,9,10,11,12])
#通过series来创建DataFrame
df = pd.DataFrame({"第一列":s1,"第二列":s2})
print(df)
#通过字典来创建DataFrame
#可以自定义索引的开始
#可以在最后修改列名来调整列的顺序
df1 = pd.DataFrame({

    "id":[1,2,3,4,5,6],
    "name":["tom","jack","david","lisa","ross","john"],
    "age":[12,13,14,15,16,17]

},index=[1,2,3,4,5,6],columns=["id","age","name"])
print(df1)
print(df1)


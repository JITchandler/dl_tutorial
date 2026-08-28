import pandas as pd

#学生分数统计
# 1. 计算每个学生的总和和平均成绩
# 2. 找出数学成绩高于90分或英语成绩高于85分的学生
# 3. 按总分从高到低排序，并输出前3名学生
data ={
    '姓名' :['张三','李四','王五','赵六','钱七'],
    '数学':[85,92,78,88,95],
    '英语':[90,88,85,92,80],
    '物理':[75,80,88,85,90]
}
scores = pd.DataFrame.from_dict(data)
scores['总分'] =scores['数学'] + scores['英语'] + scores['物理']
scores["平均分"] = scores[['数学','英语','物理']].mean(axis=1)
# res = scores[(scores['数学'] > 90) | (scores['英语'] > 85)]
res = scores.sort_values('总分', ascending=False).head(3)
# print(res)
print(res)
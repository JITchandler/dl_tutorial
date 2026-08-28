import numpy as np
import pandas as pd

# 创建一个包含10名学生数学成绩的Series，成绩范围在50-100之间。
# 计算平均分、最高分、最低分，并找出高于平均分的学生人数。
np.random.seed(42)
scores = pd.Series(np.random.randint(50, 101, 10), index=['学生'+str(i) for i in range(1, 11)])
print(scores)
print('学生平均分为:',scores.mean())
print('最高分:', scores.max())
print('最低分:', scores.min())
print('高于平均分的学生人数:',scores[scores.mean() < scores].count())

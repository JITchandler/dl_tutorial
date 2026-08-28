# 绘制 柱状图
import matplotlib.pyplot as plt
from matplotlib import  rcParams
rcParams['font.sans-serif'] = ['SimHei']

plt.title('2025年某位学生成绩统计',fontsize=20,color='red')

subjects = ['语文','数学','英语','物理','化学']
scores =   [100,110,90,80,75]
# 设置图例的时候，直接在 构造方法中设置
plt.bar(subjects,scores,width=0.8,color='blue',label = '小明')

plt.xlabel('科目名称',fontsize=10)
plt.ylabel('分数',fontsize=10)
plt.legend( loc='upper right')
# 设置每条条形的信息
for x, y in zip(subjects, scores):
    plt.text(x,y,str(y),fontsize=10)


plt.show()
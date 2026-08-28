# 绘制 饼图
import matplotlib.pyplot as plt
from matplotlib import  rcParams
rcParams['font.sans-serif'] = ['SimHei']

things = ['学习','运动','娱乐','吃饭','睡觉']
time   = [6,2,3,3,10]
colours =['#F5DEB3','#9400D3','#00FF7F','#AFEEEE','#1E90FF']
explode = [0.1,0,0,0,0]

plt.pie(
    time,
    labels=things,
    autopct="%1.1f%%",
    startangle=90,
    colors=colours,
    wedgeprops={'width':0.5},
    pctdistance=0.6,
    explode=explode
        )
plt.title('小明一天的时间分配',fontsize=20)
plt.text(0,0,'总计: \n100%',ha ='center',va='bottom',fontsize=20)
#自动优化排版
plt.tight_layout()

plt.show()
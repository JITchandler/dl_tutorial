
# 绘制折线图，展现数据的变化趋势
import matplotlib.pyplot as plt
from matplotlib import  rcParams
rcParams['font.sans-serif'] = ['SimHei']


#设置统计图的标题
plt.title('2025年月销售额统计',fontsize=20,color='red')


#设置统计图的横坐标
month = ['1月','2月','3月','4月','5月']
#设置横坐标的注释
plt.xlabel('月份',fontsize=10)

#设置统计图的纵坐标
sales = [100,300,500,200,50]
#设置纵坐标的注释
plt.ylabel('销售额',fontsize=10)

#添加左上角的图例
plt.plot(month,
         sales,
         label = '产品A',
         color = 'red',
         linestyle = '--',
         )
plt.legend(loc='upper left')
#添加网格线
plt.grid(True,alpha=0.3,linestyle='--')

#设置y轴的范围
plt.ylim(0,600)
#添加每个点的纵坐标的数据
for x , y in zip(month,sales):
    plt.text(x,y,str(y),fontsize=10)

plt.show()


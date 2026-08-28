import matplotlib.pyplot as plt
from matplotlib import  rcParams
import pandas as pd
rcParams['font.sans-serif'] = ['SimHei']
df = pd.read_csv('../data/weather.csv')
df['date']  = pd.to_datetime(df['date'])
df = df[df['date'].dt.year == 2015]
df['avg_temp'] = (df['temp_max'] + df['temp_min']) / 2
plt.plot(df['date'],
         df['temp_max'],
         label='最高温度',
         )
plt.plot(df['date'],
         df['temp_min'],
         label='最低温度',
         )
plt.plot(df['date'],
         df['avg_temp'],
         label='平均温度',
         )
plt.title("2015年气温趋势变化图")
plt.xlabel("日期")
plt.ylabel("气温")
plt.legend()

plt.show()
import numpy as np
# 温度统计
#某城市一周的最高气温(°0)为[26,30，29,31，32，30，29]。
#计算平均气温、最高气温和最低气温45找出气温超过 30°C的天数。
temp = np.array([26,30,29,31,32,30,29])
avg_temp=np.mean(temp) #平均天气
print("平均气温：",avg_temp)
max_temp=np.max(temp)
print("最高气温",max_temp)
min_temp=np.min(temp)
print("最低气温",min_temp)

days = np.sum(temp > 30)
abvoe_30 = temp[temp > 30]
print("天气大于30的天数",days)
print("具体是那一天",abvoe_30)


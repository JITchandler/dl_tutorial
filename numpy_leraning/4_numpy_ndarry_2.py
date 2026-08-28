from codecs import backslashreplace_errors

import numpy as np
# numpy的三种间隔
#等差数列
arr = np.arange(0,10,2)
print(arr)

#等间隔数列
arr1 = np.linspace(0,10,3)
print(arr1)

#等对数间隔
arr2 = np.logspace(0,4,2,base=2)
print(arr2)
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 对数据进行ADF测试
# 加载数据
data = np.loadtxt('Level_meter_data_processed.txt')

# 执行ADF测试
result = adfuller(data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 根据ADF测试结果，如果p-value > 0.05，说明数据非平稳，需要差分处理
if result[1] > 0.05:
    print("数据非平稳，考虑进行差分。")
else:
    print("数据平稳。")

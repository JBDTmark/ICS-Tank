import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# 对txt文件中的数据进行STL分解，并保存残差
# 读取数据
data = np.loadtxt('Flow_meter_data.txt')

# 创建pandas DataFrame
df = pd.DataFrame(data, columns=['Level'])

# 定义周期，根据您的数据，周期为440
period = 440

# 检查数据点是否足够多，至少需要两个周期
if len(df) < 2 * period:
    raise ValueError("数据量不足以进行STL分解。需要的最少数据点数为：{}".format(2 * period))

# 执行STL分解，保证trend和low_pass参数均大于周期且为奇数
trend = 443 if period % 2 == 0 else period + 3  # 确保trend是大于周期的奇数
low_pass = 445 if period % 2 == 0 else period + 5  # 确保low_pass大于周期且为奇数

stl = STL(df['Level'], period=period, seasonal=13, trend=trend, low_pass=low_pass, seasonal_deg=0, trend_deg=0, low_pass_deg=0)
result = stl.fit()

# 绘制分解结果
result.plot()
plt.show()

# 获取并保存残差
residuals = result.resid
np.savetxt('Flow_meter_data_residuals.txt', residuals, fmt='%f')

# 打印输出，确认结果
print("STL分解完成，残差已保存。")

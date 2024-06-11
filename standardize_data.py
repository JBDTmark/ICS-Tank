import numpy as np

def standardize_data(input_file, output_file):
    # 加载数据
    data = np.loadtxt(input_file)
    
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.std(data)
    
    # 标准化数据
    standardized_data = (data - mean) / std
    
    # 保存处理后的数据
    np.savetxt(output_file, standardized_data, fmt='%f')

# 指定输入和输出文件路径
input_file = 'Flow_meter_data_residuals.txt'  # 替换为你的残差数据文件路径
output_file = 'Flow_meter_data_standardized_residuals.txt'  # 输出文件的路径

# 执行函数
standardize_data(input_file, output_file)

import csv
import numpy as np

# 从csv文件提取指定列的数据并将其保存到txt文件中，并对数据进行标准化处理

def extract_column_to_txt(csv_file_path, column_name, txt_file_path):
    # 打开CSV文件并读取指定列
    column_data = []
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            column_value = row.get(column_name)
            # 确保值不是 None 和空字符串
            if column_value is not None and column_value != '':
                column_data.append(column_value)
    
    # 将指定列数据写入TXT文件，每行一条数据
    with open(txt_file_path, 'w') as txtfile:
        for value in column_data:
            txtfile.write(f"{value}\n")
    
    print(f"{column_name} 数据已提取到 {txt_file_path}")

# 提取数据的配置
csv_file_path = "classified_data(short).csv"
column_name = "Level meter"  # 根据你的CSV文件的列名称调整
txt_file_path = "Level_meter_data(short).txt"

# 执行提取操作
extract_column_to_txt(csv_file_path, column_name, txt_file_path)

# 数据处理部分
# 读取数据
data = np.loadtxt(txt_file_path)

# 计算均值和标准差
mean_value = np.mean(data)
std_value = np.std(data)

# 减去均值
data_centered = data - mean_value

# 标准化
data_standardized = data_centered / std_value

# 保存处理后的数据
processed_txt_file_path = "Level_meter_data_processed(short).txt"
np.savetxt(processed_txt_file_path, data_standardized, fmt='%f')
print(f"数据已处理并保存到 {processed_txt_file_path}")

import re
import csv
from collections import defaultdict

# 阅读s7comm_data.log文件并将其转换进制，然后将其保存到classified_data.log和classified_data.csv文件中
# 日志文件路径
log_file_path = "s7comm_data(short).log"

# 分类存储
data_classification = defaultdict(list)
data_entries = []

# 正则表达式模式
timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
param_item_pattern = re.compile(r'param_item: Item \[\d+\]: \((.*?)\)')
resp_data_pattern = re.compile(r'resp_data: ([\w:]+)')

# 读取并解析日志文件
with open(log_file_path, 'r') as file:
    lines = file.readlines()
    current_item = None
    current_timestamp = None

    for line in lines:
        # 查找时间戳
        timestamp_match = timestamp_pattern.search(line)
        if timestamp_match:
            current_timestamp = timestamp_match.group(1)

        # 查找param_item
        param_item_match = param_item_pattern.search(line)
        if param_item_match:
            current_item = param_item_match.group(1)
        
        # 查找resp_data
        resp_data_match = resp_data_pattern.search(line)
        if resp_data_match and current_item:
            resp_data = resp_data_match.group(1)
            data_classification[current_item].append((current_timestamp, resp_data))

# 函数：将16进制字符串转换为十进制数
def hex_to_decimal(hex_str):
    return int(hex_str, 16)

# 处理I 100数据
def process_I100_data(data):
    data = data.replace(":", "")
    if len(data) < 12:
        return None
    
    level_meter = hex_to_decimal(data[0:4])
    flow_meter = hex_to_decimal(data[4:8])
    setpoint = hex_to_decimal(data[8:12])
    
    return {
        "Level meter": level_meter,
        "Flow meter": flow_meter,
        "Setpoint": setpoint
    }

# 处理Q 100数据
def process_Q100_data(data):
    data = data.replace(":", "")
    if len(data) < 16:
        return None
    
    fill_value = hex_to_decimal(data[0:4])
    discharge_value = hex_to_decimal(data[4:8])
    sp = hex_to_decimal(data[8:12])
    pv = hex_to_decimal(data[12:16])
    
    return {
        "Fill value": fill_value,
        "Discharge value": discharge_value,
        "SP": sp,
        "PV": pv
    }

# 将分类结果保存到文件
output_file_path = "classified_data(short).log"
with open(output_file_path, 'w') as file:
    for item, data_list in data_classification.items():
        file.write(f"{item}:\n")
        for timestamp, data in data_list:
            if item.startswith("I 100"):
                processed_data = process_I100_data(data)
                if processed_data:
                    file.write(f"    Level meter: {processed_data['Level meter']}\n")
                    file.write(f"    Flow meter: {processed_data['Flow meter']}\n")
                    file.write(f"    Setpoint: {processed_data['Setpoint']}\n")
                    data_entries.append((timestamp, processed_data))
            elif item.startswith("Q 100"):
                processed_data = process_Q100_data(data)
                if processed_data:
                    file.write(f"    Fill value: {processed_data['Fill value']}\n")
                    file.write(f"    Discharge value: {processed_data['Discharge value']}\n")
                    file.write(f"    SP: {processed_data['SP']}\n")
                    file.write(f"    PV: {processed_data['PV']}\n")
                    data_entries.append((timestamp, processed_data))

# 创建CSV文件并写入数据
csv_file_path = "classified_data(short).csv"
fieldnames = ["Timestamp", "Level meter", "Flow meter", "Setpoint", "Fill value", "Discharge value", "SP", "PV"]

# 排序数据条目
data_entries.sort(key=lambda x: x[0])

with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()
    for timestamp, data in data_entries:
        row = {"Timestamp": timestamp}
        row.update({key: data.get(key, "") for key in fieldnames[1:]})
        csvwriter.writerow(row)

print(f"分类后的数据已保存到 {output_file_path} 和 {csv_file_path}")

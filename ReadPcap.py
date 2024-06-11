import pyshark
import logging
import os
# 阅读pcapng文件并解析S7comm协议数据
# 设置日志
log_file = 's7comm_data(short).log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# 解析S7comm协议数据的函数
def parse_s7comm(packet):
    try:
        if hasattr(packet, 's7comm'):
            s7comm_layer = packet.s7comm
            # 获取所有字段名称
            fields = s7comm_layer.field_names
            for field in fields:
                # 获取字段值
                field_value = getattr(s7comm_layer, field, 'N/A')
                # 记录字段名称和值
                logging.info(f'{field}: {field_value}')
            # 如果有数据字段，单独处理
            if hasattr(s7comm_layer, 'data'):
                data_value = s7comm_layer.data
                logging.info(f'Data: {data_value}')
    except Exception as e:
        logging.error(f'Error parsing packet: {e}')

# 读取pcapng文件并解析
def process_pcapng(file_path):
    try:
        capture = pyshark.FileCapture(file_path)
        for packet in capture:
            parse_s7comm(packet)
    except Exception as e:
        logging.error(f'Error reading pcapng file: {e}')

if __name__ == "__main__":
    pcapng_file = 'Data1.pcapng'
    if not os.path.isfile(pcapng_file):
        print(f'{pcapng_file} not found.')
    else:
        process_pcapng(pcapng_file)
        print(f'S7comm data logged to {log_file}')
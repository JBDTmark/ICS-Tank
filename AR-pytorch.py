import torch
import numpy as np
from torch import nn
from torch.optim import Adam

def load_data(filepath):
    # 从文本文件中读取数据
    data = np.loadtxt(filepath)
    return data

def estimate_ar_parameters(data, max_ar=256, criterion_type='AIC'):
    # 检查是否支持MPS设备并设置设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 将数据转换为torch.Tensor，并传送到设定的设备
    data = torch.tensor(data, dtype=torch.float32).to(device)
    data_len = data.shape[0]  # Use tensor shape for operations that need tensor

    # 初始化最佳模型参数
    best_criteria_value = float('inf')
    best_params = None
    best_order = None

    print(f"Starting estimation using {criterion_type}...")

    # 搜索不同的阶数
    for order in range(1, max_ar + 1):
        # 定义模型
        model = nn.Linear(order, 1, bias=False).to(device)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # 准备输入输出数据
        X = [data[i:-(order-i)] for i in range(order)]
        X = torch.stack(X, dim=-1)
        y = data[order:].unsqueeze(1)  # 确保y是二维的，与preds形状匹配

        # 训练模型
        for epoch in range(1000):  # 进行1000次迭代
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            # 打印每100次迭代的进度
            if (epoch + 1) % 100 == 0:
                print(f"{criterion_type} Order {order}: Epoch {epoch + 1}/1000, Loss: {loss.item()}")

        # 计算准则
        with torch.no_grad():
            mse = criterion(model(X), y)
            log_mse = torch.log(mse)
            if criterion_type == 'AIC':
                criteria_value = data_len * log_mse + 2 * order
            elif criterion_type == 'BIC':
                criteria_value = data_len * log_mse + order * torch.log(torch.tensor(float(data_len)).to(device))
            elif criterion_type == 'CIC':
                criteria_value = data_len * log_mse + 3 * order * torch.log(torch.tensor(float(data_len)).to(device))

        # 更新最佳准则和参数
        if criteria_value < best_criteria_value:
            best_criteria_value = criteria_value
            best_order = order
            best_params = model.weight.detach().cpu().numpy()

    return best_criteria_value, best_order, best_params

def main():
    filepath = 'Level_meter_data_standardized_residuals.txt'  # 更改为你的文件路径
    data = load_data(filepath)
    
    # 独立计算每个准则并存储结果
    aic_value, aic_order, aic_params = estimate_ar_parameters(data, max_ar=256, criterion_type='AIC')
    bic_value, bic_order, bic_params = estimate_ar_parameters(data, max_ar=256, criterion_type='BIC')
    cic_value, cic_order, cic_params = estimate_ar_parameters(data, max_ar=256, criterion_type='CIC')

    # 打印每个准则的最佳结果
    print(f"AIC Best Order: {aic_order}, Parameters: {aic_params}, AIC: {aic_value}")
    print(f"BIC Best Order: {bic_order}, Parameters: {bic_params}, BIC: {bic_value}")
    print(f"CIC Best Order: {cic_order}, Parameters: {cic_params}, CIC: {cic_value}")

if __name__ == "__main__":
    main()

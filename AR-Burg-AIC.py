import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f
from sklearn.preprocessing import StandardScaler

def read_data(file_path):
    print("Reading data from file...")
    data = pd.read_csv(file_path, header=None).squeeze()
    print("Data loaded successfully. First 10 data points:")
    print(data.head(10))
    print(f"Data statistics:\n{data.describe()}")
    return data

def burg_method(y, max_order):
    print("Executing Burg's method...")
    N = len(y)
    sigma = np.var(y)
    f = np.zeros((max_order + 1, N))
    b = np.zeros((max_order + 1, N))
    a = np.zeros((max_order + 1, max_order + 1))
    f[0, :] = y
    b[0, :] = y
    reflection_coeffs = np.zeros(max_order)
    variances = np.zeros(max_order + 1)
    variances[0] = sigma

    for k in range(1, max_order + 1):
        num = 0.0
        den = 0.0
        for t in range(k, N):
            num += f[k - 1, t] * b[k - 1, t - 1]
            den += f[k - 1, t]**2 + b[k - 1, t - 1]**2
        k_k = 2 * num / den
        reflection_coeffs[k - 1] = k_k
        print(f"Order {k}: Reflection coefficient = {k_k}")

        for t in range(k, N):
            f[k, t] = f[k - 1, t] - k_k * b[k - 1, t - 1]
            b[k, t - 1] = b[k - 1, t - 1] - k_k * f[k - 1, t]
        variances[k] = variances[k - 1] * (1 - k_k**2)
        print(f"Order {k}: Variance = {variances[k]}")

        a[k, :k] = a[k - 1, :k] - k_k * a[k - 1, k - 1::-1]
        a[k, k] = k_k
        print(f"Order {k}: AR coefficients = {a[k, :k+1]}")

    return a, reflection_coeffs, variances

def aic_criterion(variances, N):
    print("Calculating AIC values...")
    aic_values = np.zeros(len(variances))
    for p in range(1, len(variances)):
        aic_values[p] = N * np.log(variances[p]) + 2 * p
        print(f"Order {p}: AIC = {aic_values[p]}")
    best_order = np.argmin(aic_values[1:]) + 1
    print(f"Best order selected: {best_order} with AIC = {aic_values[best_order]}")
    return best_order, aic_values

def shewart_control_limits(data, control_limit_multiplier=2):
    mean = data.mean()
    std_dev = data.std()
    upper_limit = mean + control_limit_multiplier * std_dev
    lower_limit = mean - control_limit_multiplier * std_dev
    print(f"Mean: {mean}, Std Dev: {std_dev}, Upper Limit: {upper_limit}, Lower Limit: {lower_limit}")
    return upper_limit, lower_limit

def f_test(data1, data2):
    var1 = np.var(data1)
    var2 = np.var(data2)
    f_stat = var1 / var2 if var1 > var2 else var2 / var1
    p_value = 1 - f.cdf(f_stat, len(data1) - 1, len(data2) - 1)
    return f_stat, p_value

def plot_series(original, predictions, upper_limit, lower_limit, filename):
    print("Plotting data...")
    plt.figure(figsize=(12, 6))
    plt.plot(original.index, original, label='Original Data')
    plt.plot(predictions.index, predictions, label='Predictions', linestyle='--')
    plt.axhline(upper_limit, color='red', linestyle='--', label='Upper Control Limit')
    plt.axhline(lower_limit, color='green', linestyle='--', label='Lower Control Limit')
    plt.legend()
    plt.title('Original and Predicted Data with Control Limits')
    plt.savefig(filename)
    plt.show()

def main():
    file_path = 'Level_meter_data.txt'  # 更新为你的文件路径
    data = read_data(file_path)
    
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    print("Standardized data. First 10 values:")
    print(standardized_data[:10])
    print(f"Standardized Data Statistics:\nMean: {np.mean(standardized_data)}, Std Dev: {np.std(standardized_data)}, Min: {np.min(standardized_data)}, Max: {np.max(standardized_data)}")
    
    differenced_data = np.diff(standardized_data)
    print("Differenced data. First 10 values:")
    print(differenced_data[:10])
    
    max_order = 8
    a, reflection_coeffs, variances = burg_method(differenced_data, max_order)
    best_order, aic_values = aic_criterion(variances, len(differenced_data))
    print(f"Best AR model order: {best_order}")
    print("Model coefficients:", a[best_order, :best_order])

    model_coeffs = a[best_order, :best_order]
    start_point = -1000  # 从倒数第1000个数据点开始预测
    prediction_length = 400  # 预测长度为400
    predictions = np.zeros(prediction_length)
    extended_data = np.pad(differenced_data, (best_order, 0), 'constant', constant_values=(0,))
    for i in range(prediction_length):
        start_index = start_point + i - best_order
        if start_index < -len(extended_data):
            required_length = best_order
            current_data = np.concatenate((np.zeros(-start_index - len(extended_data)), extended_data[:best_order + start_index + len(extended_data)]))
        else:
            current_data = extended_data[start_index: start_index + best_order]

        if len(current_data) != best_order:
            current_data = np.pad(current_data, (0, best_order - len(current_data)), 'constant', constant_values=0)

        predictions[i] = np.dot(model_coeffs, current_data)
        if i % 100 == 0:
            print(f"Prediction step {i}: {predictions[i]}")

    last_value = standardized_data[start_point - 1]
    predictions = np.r_[last_value, predictions].cumsum()[1:]
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    print(f"Standardized Data Series Statistics:\n{pd.Series(standardized_data).describe()}")

    ucl, lcl = shewart_control_limits(pd.Series(data), control_limit_multiplier=2)  # 基于原始数据计算控制极限
    forecast_index = np.arange(len(data) + start_point, len(data) + start_point + prediction_length)
    forecast_series = pd.Series(predictions, index=forecast_index)
    
    # F检验，计算倒数1000到倒数600内的方差
    actual_data = data[start_point: start_point + prediction_length]
    f_stat, p_value = f_test(actual_data, predictions)
    print(f"F-test statistic: {f_stat}, p-value: {p_value}")
    
    # 判断预测是否良好
    if p_value > 0.05:
        print("Prediction is good.")
    else:
        print("Prediction is not good.")
    
    # 绘制截去前30000个数据的图像
    trimmed_data = data.iloc[30000:]
    trimmed_index = np.arange(30000, len(data))

    plt.figure(figsize=(12, 6))
    plt.plot(trimmed_index, trimmed_data, label='Original Data')
    plt.plot(forecast_index, predictions, label='Predictions', linestyle='--')
    plt.axhline(ucl, color='red', linestyle='--', label='Upper Control Limit')
    plt.axhline(lcl, color='green', linestyle='--', label='Lower Control Limit')
    plt.legend()
    plt.title('Original and Predicted Data with Control Limits (Trimmed)')
    plt.savefig('trimmed_forecast_plot.png')
    plt.show()

if __name__ == "__main__":
    main()

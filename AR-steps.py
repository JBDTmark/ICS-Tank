import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f
from sklearn.preprocessing import StandardScaler

def read_data(file_path):
    print("Reading data from file...")
    return pd.read_csv(file_path, header=None).squeeze()

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
        for t in range(k, N):
            f[k, t] = f[k - 1, t] - k_k * b[k - 1, t - 1]
            b[k, t - 1] = b[k - 1, t - 1] - k_k * f[k - 1, t]
        variances[k] = variances[k - 1] * (1 - k_k**2)
        a[k, :k] = a[k - 1, :k] - k_k * a[k - 1, k - 1::-1]
        a[k, k] = k_k

    return a, reflection_coeffs, variances

def aic_criterion(variances, N):
    print("Calculating AIC values...")
    aic_values = np.zeros(len(variances))
    for p in range(1, len(variances)):
        aic_values[p] = N * np.log(variances[p]) + 2 * p
    best_order = np.argmin(aic_values[1:]) + 1
    return best_order, aic_values

def shewart_control_limits(data, window_size=30):
    mean = np.mean(data[-window_size:])
    std_dev = np.std(data[-window_size:])
    upper_limit = mean + 3 * std_dev
    lower_limit = mean - 3 * std_dev
    return upper_limit, lower_limit

def f_test(var1, var2):
    return var1 / var2 if var1 > var2 else var2 / var1

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
    file_path = 'Level_meter_data.txt'  # Update this with your actual data file path
    data = read_data(file_path)
    
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    # Differencing to remove trend and seasonality
    differenced_data = np.diff(standardized_data)
    
    max_order = 10  # Adjust based on the maximum order you want to consider
    a, reflection_coeffs, variances = burg_method(differenced_data, max_order)
    best_order, aic_values = aic_criterion(variances, len(differenced_data))

    print(f"Best AR model order: {best_order}")
    print("Model coefficients:", a[best_order, :best_order])

    # Using the best AR model to predict future data
    model_coeffs = a[best_order, :best_order]
    predictions = np.zeros(400)
    
    # Extend data by padding the last value sufficiently
    extended_data = np.pad(differenced_data, (best_order, 0), 'constant', constant_values=(0,))

    for i in range(400):
        if i < best_order:
            # Use available data for the first few predictions
            predictions[i] = np.dot(model_coeffs[:i+1], extended_data[best_order+i-1::-1][:i+1])
        else:
            # Use full model order for the rest of the predictions
            predictions[i] = np.dot(model_coeffs, extended_data[best_order+i-1:best_order+i-1-best_order:-1])

    # Reverse the differencing
    last_value = standardized_data[-1]
    predictions = np.r_[last_value, predictions].cumsum()[1:]

    # Reverse the standardization
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Compute Shewart control limits
    upper_limit, lower_limit = shewart_control_limits(data)

    # Append the predictions to the data for plotting
    forecast_index = np.arange(len(data), len(data) + 400)
    forecast_series = pd.Series(predictions, index=forecast_index)

    # F-test for variance
    f_stat = f_test(np.var(data[-best_order:]), np.var(predictions[:best_order]))
    p_value = 1 - f.cdf(f_stat, best_order - 1, best_order - 1)
    print(f"F-test statistic: {f_stat}, p-value: {p_value}")

    plot_series(data, forecast_series, upper_limit, lower_limit, 'forecast_plot.png')

if __name__ == "__main__":
    main()

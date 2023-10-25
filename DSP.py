import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def mse_loss(kernel, x, y):
    smoothed_y = np.convolve(y, kernel, mode='same')
    mse = np.mean((smoothed_y - y) ** 2)
    return mse

#modify Kernel sizes
def find_optimal_kernel(x, y, kernel_sizes=[3, 5, 7, 9]):
    best_kernel = None
    best_mse = float('inf')

    for size in kernel_sizes:
        initial_kernel = np.ones(size) / size
        result = minimize(mse_loss, initial_kernel, args=(x, y), method='Nelder-Mead')

        if result.fun < best_mse:
            best_mse = result.fun
            best_kernel = result.x

    return best_kernel

#modify Kernel sizes and num_folds (cross-validation)
def find_optimal_parameters(x, y, method='savgol', num_folds=2, kernel_sizes=[1,3,5]):
    if method == 'custom':
        optimal_kernel = find_optimal_kernel(x, y, kernel_sizes)
        return {'method': method, 'optimal_kernel': optimal_kernel}

    folds = np.array_split(np.arange(len(x)), num_folds)
    avg_errors = []

    for param in range(2, 12, 2):  # Trying different parameter values
        errors = []
        for i in range(num_folds):
            train_idx = np.concatenate([folds[j] for j in range(num_folds) if j != i])
            test_idx = folds[i]
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]

            if method == 'savgol':
                smoothed_y = savgol_filter(y_train, param, 2)
            elif method == 'gaussian':
                smoothed_y = gaussian_filter1d(y_train, param)
            elif method == 'bin':
                _, bin_edges, _ = np.histogram(x_train, bins=param)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                smoothed_y = np.histogram(x_train, bins=param, weights=y_train)[0] / np.histogram(x_train, bins=param)[
                    0]
            else:
                raise ValueError("Invalid smoothing method. Options: 'savgol', 'gaussian', 'bin', 'custom'.")

            errors.append(np.mean((smoothed_y - y_test) ** 2))
        avg_errors.append(np.mean(errors))

    optimal_param = (np.arange(2, 12, 2))[np.argmin(avg_errors)]
    return {'method': method, 'optimal_param': optimal_param}

#modify Kernel sizes and num_folds (cross-validation)
def smooth_data_with_optimal_params(x, y, method='savgol', num_folds=2, kernel_sizes=[3, 5, 7, 9]):
    if method == 'custom':
        optimal_kernel = find_optimal_kernel(x, y, kernel_sizes)
        smoothed_y = np.convolve(y, optimal_kernel, mode='same')
    else:
        optimal_params = find_optimal_parameters(x, y, method, num_folds, kernel_sizes)
        optimal_param_value = optimal_params['optimal_param']

        if method == 'savgol':
            smoothed_y = savgol_filter(y, optimal_param_value, 2)
        elif method == 'gaussian':
            smoothed_y = gaussian_filter1d(y, optimal_param_value)
        elif method == 'bin':
            _, bin_edges, _ = np.histogram(x, bins=optimal_param_value)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            smoothed_y = np.histogram(x, bins=optimal_param_value, weights=y)[0] / \
                         np.histogram(x, bins=optimal_param_value)[0]
        else:
            raise ValueError("Invalid smoothing method. Options: 'savgol', 'gaussian', 'bin', 'custom'.")

    return smoothed_y

# Generate random noisy data
np.random.seed(0)  # Setting seed for reproducibility
x = np.linspace(0, 10, 10000)
y_noisy = np.sin(x) + np.random.normal(0, 0.3, len(x))  # Sine wave with noise

# Apply optimal smoothing using the previously defined function
smoothed_data = smooth_data_with_optimal_params(x, y_noisy, method='gaussian')

# Plot the unsmoothed and smoothed data
plt.figure(figsize=(8, 6))
plt.scatter(x, y_noisy, color='blue', label='Noisy Data')
plt.plot(x, smoothed_data, color='red', label='Smoothed Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Noisy Data vs. Smoothed Data')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold


def find_optimal_kernel(x, y, kernel_sizes=[3, 5, 7, 9], num_folds=5):
    best_kernel = None
    best_mse = float('inf')

    kf = KFold(n_splits=num_folds)

    for size in kernel_sizes:
        mse_scores = []
        for train_idx, test_idx in kf.split(x):
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]

            initial_kernel = np.ones(size) / size
            result = minimize(mse_loss, initial_kernel,
                              args=(x_train, y_train), method='Nelder-Mead')

            smoothed_y = np.convolve(y_test, result.x, mode='same')
            mse = np.mean((smoothed_y - y_test) ** 2)
            mse_scores.append(mse)

        avg_mse = np.mean(mse_scores)

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_kernel = result.x

    return best_kernel


def mse_loss(kernel, x, y):
    pad_width = len(kernel) // 2  # Assuming kernel size is odd for symmetric padding
    y_padded = np.pad(y, pad_width, mode='symmetric')
    smoothed_y = np.convolve(y_padded, kernel, mode='valid')
    mse = np.mean((smoothed_y - y) ** 2)
    return mse


def optimal_num_bins(x, y):
    data_range = np.ptp(x)
    q75, q25 = np.percentile(y, [75, 25])
    iqr = q75 - q25
    h = 2 * iqr / (len(y) ** (1/3))
    num_bins = int(data_range / h)
    return num_bins


# modify Kernel sizes and num_folds (cross-validation)
def find_optimal_parameters(x, y, method='savgol',
                            num_folds=2, kernel_sizes=[1, 3, 5]):
    if method == 'custom':
        optimal_kernel = find_optimal_kernel(x, y, kernel_sizes)
        return {'method': method, 'optimal_kernel': optimal_kernel}

    folds = np.array_split(np.arange(len(x)), num_folds)
    avg_errors = []

    optimal_param_value = None
    # Initialize optimal_param_value outside the method blocks
    if method == 'bin':
        optimal_param_value = optimal_num_bins(x, y)
        hist, bin_edges = np.histogram(x, bins=optimal_param_value)
        return {'method': method, 'optimal_param': bin_edges}

    for param in range(3, 6, 2):  # Trying different parameter values 12
        errors = []
        for i in range(num_folds):
            train_idx = np.concatenate([folds[j]
                                        for j in range(num_folds) if j != i])
            test_idx = folds[i]
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]

            if method == 'savgol':
                smoothed_y = savgol_filter(y_train, param, 2)
            elif method == 'gaussian':
                smoothed_y = gaussian_filter1d(y_train, param)
            else:
                raise ValueError("Invalid smoothing method. " +
                                 "Options: 'savgol', " +
                                 "'gaussian', 'bin', 'custom'.")

            errors.append(np.mean((smoothed_y - y_test) ** 2))
        avg_errors.append(np.mean(errors))

    optimal_param_index = np.argmin(avg_errors)
    optimal_param_value = 3 + 2 * optimal_param_index
    # Calculate the optimal param value
    return {'method': method, 'optimal_param': optimal_param_value}
# modify Kernel sizes and num_folds (cross-validation)


def smooth_data_with_optimal_params(x, y, method='savgol',
                                    num_folds=2, kernel_sizes=[3, 5, 7, 9]):
    if method == 'custom':
        optimal_kernel = find_optimal_kernel(x, y, kernel_sizes)
        smoothed_y = np.convolve(y, optimal_kernel, mode='same')
    else:
        optimal_params = find_optimal_parameters(x, y, method,
                                                 num_folds, kernel_sizes)
        optimal_param_value = optimal_params['optimal_param']

        if method == 'savgol':
            smoothed_y = savgol_filter(y, optimal_param_value, 2)
        elif method == 'gaussian':
            smoothed_y = gaussian_filter1d(y, optimal_param_value)
        elif method == 'bin':
            hist, bin_edges = np.histogram(x, bins=optimal_param_value)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            smoothed_y = np.histogram(x, bins=bin_edges, weights=y)[0] / hist
            # Interpolate smoothed_y to match the length of x
            interp_func = interp1d(bin_centers, smoothed_y,
                                   kind='linear', fill_value='extrapolate')
            smoothed_y = interp_func(x)
        else:
            raise ValueError("Invalid smoothing method. " +
                             "Options: 'savgol', " +
                             "'gaussian', 'bin', 'custom'.")

    return smoothed_y

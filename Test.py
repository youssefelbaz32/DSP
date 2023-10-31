import numpy as np
import pytest
import DSP

# Test find_optimal_kernel function
def test_find_optimal_kernel():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    kernel = DSP.find_optimal_kernel(x, y)
    assert len(kernel) > 0

# Test optimal_num_bins function
def test_optimal_num_bins():
    x = np.random.rand(100)
    y = np.random.rand(100)
    num_bins = DSP.optimal_num_bins(x, y)
    assert num_bins > 0

# Test find_optimal_parameters function
def test_find_optimal_parameters():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    optimal_params = DSP.find_optimal_parameters(x, y, method='savgol')
    assert 'method' in optimal_params
    assert 'optimal_param' in optimal_params

# Test smooth_data_with_optimal_params function
def test_smooth_data_with_optimal_params():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.3, len(x))
    smoothed_data = DSP.smooth_data_with_optimal_params(x, y, method='savgol')
    assert len(smoothed_data) == len(x)

def test_smooth_data_with_optimal_params_custom():
    # Test custom kernel smoothing
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 2, 1])
    smoothed_custom = DSP.smooth_data_with_optimal_params(x, y, method='custom')
    expected_output = np.array([0.99999183, 1.99998708, 2.9999419 , 1.99999391, 0.99999867])
    assert np.allclose(smoothed_custom, expected_output,rtol=1e-6)

def test_smooth_data_with_optimal_params_savgol():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Larger input data
    y = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1, 2])  # Corresponding y values
    smoothed_savgol = DSP.smooth_data_with_optimal_params(x, y, method='savgol')
    expected_output = np.array([0.94285714,2.22857143,2.65714286,2.,1.34285714, 2., 2.65714286, 2.,1.8,1.6])
    assert np.allclose(smoothed_savgol, expected_output, rtol=1e-6)

def test_smooth_data_with_optimal_params_bin():
    # Test binning smoothing
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 2, 1])
    smoothed_bin = DSP.smooth_data_with_optimal_params(x, y, method='bin')
    expected_output = np.array([0.75, 1.875, 3, 1.875, 0.75])
    assert np.allclose(smoothed_bin, expected_output)

# Run tests with pytest
if __name__ == "__main__":
    pytest.main()

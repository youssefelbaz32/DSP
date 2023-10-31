import numpy as np
import matplotlib.pyplot as plt
import DSP

# Generate random noisy data
np.random.seed(0)  # Setting seed for reproducibility
x = np.linspace(0, 10, 1000)
y_noisy = np.sin(x) + np.random.normal(0, 0.3, len(x))  # Sine wave with noise

# Apply optimal smoothing using the previously defined function
smoothed_data = DSP.smooth_data_with_optimal_params(x, y_noisy, method='savgol')

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

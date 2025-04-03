import numpy as np
import matplotlib.pyplot as plt
import padasip as pa

# Set random seed for reproducibility.
np.random.seed(42)

# Number of data points and filter order (input dimension)
N = 200
n_features = 2

# True coefficients for the linear system
true_w = np.array([2.0, -3.0])

# Generate random input data (each row is an input vector)
x = np.random.randn(N, n_features)

# Generate desired output: d = true_w^T * x + noise
noise = 1 * np.random.randn(N)
d = np.dot(x, true_w) + noise

# Initialize the RLS filter from padasip:
# lambda_ is the forgetting factor (typically close to 1), and
# delta is used to initialize the covariance matrix (P(0) = I/delta).
rls = pa.filters.FilterRLS(n=n_features, mu=0.99)

# Run the RLS filter on the data.
# The run() method returns:
#  y: predictions, e: error at each step, w: array of estimated coefficients over time.
y, e, w = rls.run(d, x)

# Plot the evolution of the filter coefficients
plt.figure(figsize=(10, 5))
plt.plot(w[:, 0], label='Estimated w[0]')
plt.plot(w[:, 1], label='Estimated w[1]')
plt.hlines(true_w[0], 0, N, colors='blue', linestyles='dashed', label='True w[0]')
plt.hlines(true_w[1], 0, N, colors='orange', linestyles='dashed', label='True w[1]')
plt.xlabel('Iteration')
plt.ylabel('Coefficient value')
plt.title('RLS Filter Coefficient Convergence using padasip')
plt.legend()
plt.show()

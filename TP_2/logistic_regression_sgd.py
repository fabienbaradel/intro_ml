import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Create data - 2D points
N = 100
x1 = np.random.uniform(0, 1, N)
x2 = np.random.uniform(0, 1, N)
y = np.asarray([0 if a + b > 1 else 1 for a, b in zip(x1, x2)])
x = np.column_stack([x1, x2])  # (N,2)

# Plot
plot_color = ['r' if c == 0 else 'g' for c in y]
plt.scatter(x[:, 0], x[:, 1], color=plot_color, label='Original data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
plt.clf()



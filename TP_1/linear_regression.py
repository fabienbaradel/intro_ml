import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from numpy import dot, transpose
from numpy.linalg import inv

# Generate Data: y = 1.5*x + 5 with x between 20 and 40
min_x, max_x = 20, 40
list_x = np.random.uniform(min_x, max_x, 100).tolist()
list_x.sort()
f = lambda x_i: 1.5 * x_i + 5
SCALE = 5
list_y = [f(x) + np.random.normal(0, SCALE, 1)[0] for x in list_x]

plt.plot(list_x, list_y, 'ro', label='Original Data')
plt.xlabel('x')
plt.ylabel('y')
# plt.show() # for showing the data points


#### ----- CLOSED-FORM SOLUTION ----- ####

# Add 1 to x
ones = np.ones(len(list_x))  # (N,)
X = np.stack((ones, list_x), axis=1)  # (N, 2)
Y = np.asarray(list_y)  # (N,)

# Computing Beta
Xt = transpose(X)
product = dot(Xt, X)
theInverse = inv(product)
Beta = dot(dot(theInverse, Xt), list_y)
b_cf, w_cf = Beta[0], Beta[1]
print(f"CLOSED-FORM: w={w_cf:.2f} b={b_cf:.2f}\n")

# Plot fitted line
plt.plot([min_x, max_x], [b_cf + w_cf * min_x, b_cf + w_cf * max_x],
         '-', markersize=3, color='blue', label='Closed-form')


# plt.show() # for showing the data points


#### ----- SIMPLE LINEAR REGRESSION BY GD ----- ####

def loss(list_x, list_y, w, b):
    """ least square loss """
    N = len(list_y)
    total_error = 0.0
    for i in range(N):
        total_error += (list_y[i] - (list_x[i] * w + b)) ** 2
    return total_error / float(N)


loss([1], [3], 1, 2)  # 0


def predict(x, w, b):
    return w * x * b


predict(5, 2, -1)  # 9


def dl_dw(x, y, w, b):
    return -2 * x * (y - (w * x + b))


dl_dw(4, -1, 0, 0)


def dl_db(x, y, w, b):
    return -2 * (y - (w * x + b))


def update_w_and_b(list_x, list_y, w, b, lr):
    grad_w, grad_b = 0.0, 0.0
    N = len(list_y)

    for i in range(N):
        grad_w += dl_dw(list_x[i], list_y[i], w, b)
        grad_b += dl_db(list_x[i], list_y[i], w, b)

    # update
    w -= (1 / float(N)) * grad_w * lr
    b -= (1 / float(N)) * grad_b * lr

    return w, b


# Hyper-Param
w_init, b_init = 2., 4.
# w_init, b_init = 0., 0.
w, b = w_init, b_init
NB_ITER = 5000
LR = 0.001
N = len(list_y)

# --- Gradient Descent solution ----- #
for i in range(NB_ITER):
    # update w and b
    w, b = update_w_and_b(list_x, list_y, w, b, LR)

    # Log
    if i % 1000 == 0:
        print(f"Iter: {i} \t Loss = {loss(list_x, list_y, w, b):4f} \t w = {w:2f} \t b = {b:.2f}")

# Store values
w_gd, b_gd = w, b
print(f"GRADIENT DESCENT: w={w_gd:.2f} b={b_gd:.2f}\n")

# Plot fitted line
plt.plot([min_x, max_x], [b_gd + w_gd * min_x, b_gd + w_gd * max_x],
         '-', markersize=3, color='orange', label='Gradient Descent')

# --- Stochastic Gradient Descent solution ----- #
BATCH_SIZE = 4
for i in range(NB_ITER):
    # sample a subset of data points
    idx_batch = list(np.random.choice(range(N), BATCH_SIZE))
    list_x_batch = [list_x[idx] for idx in idx_batch]
    list_y_batch = [list_y[idx] for idx in idx_batch]

    # update w and b
    w, b = update_w_and_b(list_x_batch, list_y_batch, w, b, LR)

    # Log
    if i % 1000 == 0:
        print(f"Iter: {i} \t Loss = {loss(list_x, list_y, w, b):4f} \t w = {w:2f} \t b = {b:.2f}")

# Store values
w_sgd, b_sgd = w, b
print(f"STOCHASTIC GRADIENT DESCENT: w={w_sgd:.2f} b={b_sgd:.2f}\n")

# Plot fitted line
plt.plot([min_x, max_x], [b_sgd + w_sgd * min_x, b_sgd + w_sgd * max_x],
         '-', markersize=3, color='green', label='Stochastic Gradient Descent')

## -- SKLEARN ---- ##
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(np.asarray(list_x).reshape(-1, 1), list_y)
w_sklearn, b_sklearn = model.coef_[0], model.intercept_
print(f"SKLEARN: w={w_sklearn:.2f} b={b_sklearn:.2f}\n")
# Plot fitted line
plt.plot([min_x, max_x], [b_sklearn + w_sklearn * min_x, b_sklearn + w_sklearn * max_x],
         '-', markersize=3, color='yellow', label='Sklearn')
plt.legend(loc=0)
plt.show()
plt.clf()

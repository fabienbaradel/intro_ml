import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.use("TkAgg")  # only if Mac OS
import matplotlib.pyplot as plt

# Question 1: Load the dataset and split examples into train/val/test sets
boston = load_boston()  # a dict
X, target = boston['data'], boston['target']
N = len(boston['target'])  # total number of examples
idx_train = list(range(300))  # 300 first examples
idx_val = list(range(300, 400))  # 100 next examples
idx_test = list(range(400, N))  # next examples

# Question 2: Plot y vs CRIM
crim = X[:, 0].reshape(-1, 1)  # take only the variable CRIM


def plot_datapoints(x, y, xlabel='x', ylabel='y', color='red'):
    """ plot the datapoints """
    plt.scatter(x, y, c=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# Plot (y,crim) of the training set
plot_datapoints(crim[idx_train], target[idx_train], xlabel='crim')
plt.show()
plt.clf()  # reset the plot

# Question 3 : Build a linear regression model: target = w . crim + b
# and plot predictions vs ground-truths
model = LinearRegression()
model.fit(crim[idx_train], target[idx_train])  # take only training examples for estimating w and b
print(f"Our model: target = {model.coef_[0]} . crim + {model.intercept_}")

# plot true points vs estimated points
preds_train = model.predict(crim[idx_train])
plot_datapoints(crim[idx_train], preds_train, color='red')  # (x,\hat{y})
plot_datapoints(crim[idx_train], target[idx_train], xlabel='CRIM', ylabel='target', color='green')  # (x,y)
plt.show()
plt.clf()  # reset the plot


# Question 4: Write a function which is computing the Mean Square Error given prediction and ground-truth
def mean_square_error(preds, y):
    errors = (preds - y) ** 2
    return np.mean(errors)


# MSE on training set
mse_train = mean_square_error(preds_train, target[idx_train])
# MSE on validation set
preds_val = model.predict(crim[idx_val])  # prediction on validation set -- with the idx_val
mse_val = mean_square_error(preds_val, target[idx_val])
print(f"MSE train = {mse_train} \t MSE val = {mse_val}")

# Question 5 : Build a linear regression model: log(target) = w . crim + b
# plot
log_target = np.log(target)
plot_datapoints(crim[idx_train], log_target[idx_train], xlabel='crim', ylabel='log(target')
plt.show()
plt.clf()  # reset the plot

# model
model = LinearRegression()
model.fit(crim[idx_train], log_target[idx_train])
print(f"Our model: log(target) = {model.coef_[0]} . crim + {model.intercept_}")

# MSE on training set
preds_train = np.exp(model.predict(crim[idx_train]))  # because exp(log(y)) = y
mse_train = mean_square_error(preds_train, target[idx_train])
# MSE on validation set
preds_val = np.exp(model.predict(crim[idx_val]))  # prediction on validation set -- with the idx_val
mse_val = mean_square_error(preds_val, target[idx_val])
print(f"MSE train = {mse_train} \t MSE val = {mse_val}")

# Question 6: Write a function which takes which given the variables and the target, train and compute MSE
identity = lambda x: x  # default function for normalization - doing nothing


def train_and_metrics(X, y, idx_train, idx_val,
                      norm_x=identity, norm_y=identity, denorm_y=identity):
    # Create the model
    model = LinearRegression()
    model.fit(norm_x(X[idx_train]), norm_y(y[idx_train]))  # normalization of X and y
    print(f"W={model.coef_}\tB={model.intercept_}")

    # Predictions
    preds_train = denorm_y(model.predict(X[idx_train]))  # denormalization
    preds_val = denorm_y(model.predict(X[idx_val]))

    # Compute MSE
    mse_train = mean_square_error(preds_train, y[idx_train])
    mse_val = mean_square_error(preds_val, y[idx_val])

    return mse_train, mse_val


mse_train, mse_val = train_and_metrics(crim, target, idx_train, idx_val, norm_y=np.log, denorm_y=np.exp)
print(f"MSE train = {mse_train} \t MSE val = {mse_val}")

# Question 7: Build a linear regression model log(target) = w . log(crim) + b
mse_train, mse_val = train_and_metrics(crim, target, idx_train, idx_val, norm_x=np.log, norm_y=np.log, denorm_y=np.exp)
print(f"MSE train = {mse_train} \t MSE val = {mse_val}")


# Question 8: create a function which is doing min-max normlization and build the model: log(target) = w . min_max_norm(crim) + b
def min_max_norm(x):
    return (x - np.min(x) / (np.max(x) - np.min(x)))


mse_train, mse_val = train_and_metrics(crim, target, idx_train, idx_val, norm_x=min_max_norm, norm_y=np.log,
                                       denorm_y=np.exp)
print(f"MSE train = {mse_train} \t MSE val = {mse_val}")

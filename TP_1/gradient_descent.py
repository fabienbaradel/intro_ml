""" Gradient Descent in python """
import numpy as np
import matplotlib
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot_function(X, J_theta, show=False):
    # Plot J(x) vs x
    plt.plot(X, J_theta, label='Fitted line - closed form')
    plt.xlabel('x')
    plt.ylabel('J(x)')

    if show:
        plt.show()

        # point
        # x_0=-1
        # plt.plot(x_0, J(x_0), marker='o', markersize=5, color="red")


def gradient_descent(theta_0, lr, nb_iters, dJ_dtheta, J):
    """ implementation of the gradient descent """
    # legend
    plt.text(0.0, 90, f"theta_0={theta_0} lr={lr}", fontsize=12)

    # init
    theta = theta_0
    for t in range(nb_iters):

        # Print in the log
        if t % 10 == 0:
            print(f"Iter: {t} \t J(x) = {J(theta):2f} \t x = {theta:2f}")

        # Update theta
        prev_theta = theta  # for visu
        theta -= lr * dJ_dtheta(theta)

        # Vizualization of the move from theta_t to theta_{t+1}
        plt.plot([theta, prev_theta], [J(theta), J(prev_theta)], 'ro-', markersize=3, color='red')

    # Solution
    print(f"Minimum local found at: {theta:2f}")


# Function J(theta) = (theta+1)^2
J = lambda theta_i: (theta_i + 1) ** 2

# Generate data
list_theta = np.arange(-10, 10, 0.01).tolist()
list_J_theta = [J(theta_i) for theta_i in list_theta]

# Plot function
plot_function(list_theta, list_J_theta, show=False)  # should be True if you want to plot
# plt.clf()  # remove everything in the plot

# Derivative dJ/dtheta
dJ_dtheta = lambda x: 2 * (x + 1)

# Run the gradient descent on the same function with different parameters
dir_png = 'log/gd'
os.makedirs(f"./{dir_png}", exist_ok=True)

nb_iters = 100
for theta_0 in [-8, -1, -7]:
    for lr in [0.1, 0.01, 0.001, -0.01, 0.8, 1.01]:
        print(f"\nx_0={theta_0} lr={lr}")
        # Plot the function
        plot_function(list_theta, list_J_theta, show=False)

        # Find a local minimum
        gradient_descent(theta_0=theta_0, lr=lr, nb_iters=nb_iters, dJ_dtheta=dJ_dtheta, J=J)

        # Save the plot into a file
        plt.savefig(f"./{dir_png}/x_0={theta_0:.1f}_lr={lr:.3f}.png")

        # Clears plot
        plt.clf()

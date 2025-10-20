# Author: Kenji Kashima
# Date  : 2025/09/01
# Note  : pip install cvxpy

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import Variable, Minimize, quad_form, Problem
from scipy.stats import truncnorm, laplace
import config

np.random.seed(1)

# System matrices and parameters setup
def initialize_matrices():
    k_bar = 60  # total steps

    # System matrices
    A = np.array([[-0.39, -0.67, -0.34],
                  [0.71, -0.51, 0.11],
                  [-0.46, -0.35, -0.12]])  # System matrix A
    B_v = np.array([[0], [1], [0]])  # Input matrix B_v
    C = np.array([[1, 0, 0]])  # Output matrix C

    # Initial state mean
    mu_0 = np.array([1, 1, 1])  # Mean of the initial state
    sigma_0 = np.eye(3)  # Covariance of the initial state

    return A, B_v, C, mu_0, sigma_0, k_bar

# Generate system dynamics and noise
def generate_data(A, B_v, C, mu_0, sigma_0, k_bar):

    x_dim = A.shape[0]
    # Noise parameters
    v_bounds = [-1, 1]  # Range for process noise v_k (truncated Gaussian)
    laplace_scale = 1  # Scale parameter for Laplace noise w_k

    # Generate the data sequence for x and y
    x = np.random.multivariate_normal(mu_0, sigma_0)  # Initial state x_0 ~ N(mu_0, I)
    x_data = np.zeros((x_dim, k_bar+1))  # Store x_{0:k_bar}
    y_data = np.zeros(k_bar+1)  # Store y_{0:k_bar}

    x_data[:, 0] = x

    # Process noise and measurement noise
    v_data = []
    w_data = []

    for k in range(k_bar):
        # Generate truncated standard Gaussian noise v_k bounded in [-1, 1]
        v_k = truncnorm.rvs(v_bounds[0], v_bounds[1], loc=0, scale=1)

        # Generate Laplace distributed noise w_k ~ Lap(0, 1)
        w_k = laplace.rvs(scale=laplace_scale)

        # Update the system dynamics
        y_data[k] = (C @ x).item() + w_k  # Output measurement y_k
        x = A @ x.reshape(x_dim, 1) + B_v * v_k  # State update x_{k+1}

        # Store the values for future use
        x_data[:, k+1] = x.flatten()
        v_data.append(v_k)
        w_data.append(w_k)

    # The last measurement noise w_{k_bar}
    w_k = laplace.rvs(scale=laplace_scale)
    w_data.append(w_k)
    y_data[k_bar] = (C @ x).item() + w_k

    return x_data, y_data, v_data, w_data

# Solve the optimization problem using cvxpy
def solve_optimization(A, B_v, C, mu_0, k_bar, y_data):
    x_dim = A.shape[0]
    # Define optimization variables
    x0 = Variable(x_dim)
    u = Variable(k_bar)     
    t = Variable(k_bar + 1)      # | xhat - y_data[i] |

    # Quadratic objective
    Q_x0 = np.eye(x_dim)
    Q_u = np.eye(k_bar)

    # Objective: 1/2 * ||x0 - mu0||^2 + sum t_i
    objective = Minimize(quad_form(x0 - mu_0, Q_x0)/2 + quad_form(u,Q_u)/2 + sum(t))

    # Constraints: L1 norm and control input bounds
    constraints = []

    # Aux1 @ x0_est + Aux2 @ u_est = C xhat
    Aux1 = np.zeros((k_bar + 1, x_dim))
    Aux2 = np.zeros((k_bar + 1, k_bar))
    for i in range(k_bar + 1): 
        Aux1[i, :] = (C @ np.linalg.matrix_power(A, i)).ravel()
        for j in range(i):   # j=0..i-1
            Aux2[i, j] = (C @ np.linalg.matrix_power(A, i-1-j) @ B_v).item()

    # L1 norm constraints     
    for i in range(k_bar + 1):
        constraints += [Aux1[i, :] @ x0 + Aux2[i, :] @ u <= y_data[i] + t[i],
                        -Aux1[i, :] @ x0 - Aux2[i, :] @ u <= -y_data[i] + t[i]]

    # Control input bounds |u_k| <= 1
    for k in range(k_bar):
        constraints += [u[k] <= 1, u[k] >= -1]

    # Solve problem
    prob = Problem(objective, constraints)
    prob.solve(verbose=True)

    # Recover state estimates
    xhat = np.zeros((x_dim, k_bar + 1))
    xhat[:, 0] = x0.value
    for k in range(k_bar):
        xhat[:, k+1] = (A @ xhat[:, k].reshape(x_dim,1) + B_v * u.value[k]).ravel()

    return u.value, xhat

# Plot figures
def plot_figures(x_data, y_data, v_data, w_data, u_est, xhat, k_bar):
    x_max = 6
    def figure5_4a():
        figsize = config.global_config(type=0)
        plt.figure(figsize=figsize)
        plt.plot(range(k_bar+1), x_data[0, :], label=r'True state $(x_k)_1$', linewidth=1.5)
        plt.plot(range(k_bar+1), y_data, 'g--', label=r'Measurements $y$', linewidth=1.5)
        plt.plot(range(k_bar+1), xhat[0, :], 'r-.', label=r'Estimated $(\hat{x}_k)_1$', linewidth=1.5)
        plt.xlabel(r'$k$')
        plt.legend()
        plt.grid(True)
        plt.xlim([0,k_bar])
        plt.ylim([-x_max,x_max])
        plt.tight_layout()
        plt.savefig("./Figure5_4a.pdf")
        plt.show()

    def figure5_4b():
        figsize = config.global_config(type=0)
        plt.figure(figsize=figsize)
        plt.plot(range(k_bar+1), x_data[2, :], label=r'True state $(x_k)_3$', linewidth=1.5)
        plt.plot(range(k_bar+1), xhat[2, :], 'r-.', label=r'Estimated $(\hat{x}_k)_3$', linewidth=1.5)
        plt.xlabel(r'$k$')
        plt.legend()
        plt.grid(True)
        plt.xlim([0,k_bar])
        plt.ylim([-x_max,x_max])
        plt.tight_layout()
        plt.savefig("./Figure5_4b.pdf")
        plt.show()

    def figure5_4c():
        figsize = config.global_config(type=0)
        plt.figure(figsize=figsize)
        plt.step(range(k_bar), v_data, label=r'Disturbance $v_k$', linewidth=1.5)
        plt.step(range(k_bar), u_est, 'r-.', label=r'Estimated $\hat{v}_k:=u(k)$', linewidth=1.5)
        plt.xlabel(r'$k$')
        plt.legend()
        plt.grid(True)
        plt.xlim([0,k_bar-1])
        plt.ylim([-x_max,x_max])
        plt.tight_layout()
        plt.savefig("./Figure5_4c.pdf")
        plt.show()

    def figure5_4d():
        figsize = config.global_config(type=0)
        plt.figure(figsize=figsize)
        plt.step(range(k_bar+1), w_data, label=r'Noise $w_k$', linewidth=1.5)
        plt.step(range(k_bar+1), y_data - xhat[0, :], 'r-.', label=r'Estimated $\hat{w}_k$', linewidth=1.5)
        plt.xlabel(r'$k$')
        plt.legend()
        plt.grid(True)
        plt.xlim([0,k_bar])
        plt.ylim([-x_max,x_max])
        plt.tight_layout()
        plt.savefig("./Figure5_4d.pdf")
        plt.show()

    figure5_4a()
    figure5_4b()
    figure5_4c()
    figure5_4d()

# Main function to run the simulation and plotting
def main():
    A, B_v, C, mu_0, sigma_0, k_bar = initialize_matrices()
    x_data, y_data, v_data, w_data = generate_data(A, B_v, C, mu_0, sigma_0, k_bar)
    u_est, xhat = solve_optimization(A, B_v, C, mu_0, k_bar, y_data)
    plot_figures(x_data, y_data, v_data, w_data, u_est, xhat, k_bar)

if __name__ == '__main__':
    main()


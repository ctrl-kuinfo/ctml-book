# Author: Kenji Kashima
# Date  : 2025/09/01
# Note  : pip install cvxpy

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_ellipse(Sigma):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    
    # Calculate rotation angle in radians
    rotation_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Calculate the lengths of the axes
    axes_lengths = 2 * np.sqrt(eigenvalues)

    # Generate theta values
    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Calculate the x and y coordinates of the ellipse
    x = axes_lengths[0] * np.cos(theta) * np.cos(rotation_rad) - axes_lengths[1] * np.sin(theta) * np.sin(rotation_rad)
    y = axes_lengths[0] * np.cos(theta) * np.sin(rotation_rad) + axes_lengths[1] * np.sin(theta) * np.cos(rotation_rad)
    
    return x, y

def figure11_1(label="a"):
    figsize = config.global_config(type= 1)
    # Define parameters
    A = np.array([[1, 0.1], [-0.3, 1]])  # State matrix
    B = np.array([[0.7], [0.4]])         # Control matrix
    N = 0.1 * np.eye(2)                  # Noise covariance matrix
    x_dim, u_dim = B.shape
    Sigma_0 =  3 * np.eye(x_dim)  # Initial covariance
    Sigma_10 = np.array([[2, 0], [0, 0.5]])  # Target covariance
    k_bar = 10  # Time horizon

    # Define optimization variables
    Sigma = [cp.Variable((x_dim, x_dim), PSD=True) for _ in range(k_bar+1)]  # Sigma_k, k=0 to k_bar
    P = [cp.Variable((x_dim,u_dim)) for _ in range(k_bar)]  # P_k, k=0 to k_bar-1
    M = [cp.Variable((u_dim,u_dim)) for _ in range(k_bar)]  # M_k, k=0 to k_bar-1, non-negative

    # Initial and terminal conditions
    constraints = [Sigma[0] == Sigma_0 ]  # Sigma_0
    constraints += [Sigma[k_bar] == Sigma_10]  # Sigma_10
    if label!="a":
        constraints += [Sigma[5][1,1] <= 0.5]

    # Recurrence relation constraints
    for k in range(k_bar):
        # Sigma_{k+1} = A * Sigma_k * A' + A * P_k * B' + B * P_k' * A' + B * M_k * B' + N
        Sigma_next = A @ Sigma[k] @ A.T + A @ P[k] @ B.T + B @ P[k].T @ A.T + B @ M[k] @ B.T + N
        constraints += [Sigma[k+1] == Sigma_next]
        
        # Semi-definite constraint: [Sigma_k, P_k; P_k', M_k] >= 0
        LMI = cp.bmat([[Sigma[k], P[k]], [P[k].T, M[k]]])
        constraints += [LMI >> 0]  # Ensure LMI is positive semi-definite

    # Objective function: minimize trace(sum(M_k))
    objective = cp.Minimize(cp.trace(cp.sum(M)))

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)  # You can use SCS or other SDP solvers

    # Check the feasibility and output results
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        opt_value = problem.value
        print(label + " Optimal value:", opt_value)
        Sigma_optimal = [Sigma[k].value for k in range(k_bar+1)]
        P_optimal = [P[k].value for k in range(k_bar)]
        M_optimal = [M[k].value for k in range(k_bar)]
        print(label + " Sigma_opt_5(2,2):", Sigma_optimal[5][1,1])

        # Number of trajectories
        num_trajectories = 20
        trajectories = []

        for _ in range(num_trajectories):
            # Generate random initial state x_0 ~ N(0, 3I)
            x = np.random.multivariate_normal(mean=np.zeros(x_dim), cov=Sigma_0)
            trajectory = [x]
        
            for k in range(k_bar):
                K_k = np.linalg.solve(Sigma_optimal[k], P_optimal[k]).T  # Compute optimal gain K_k
                v_k = np.random.multivariate_normal(mean=np.zeros(x_dim), cov=N)  # Noise v_k ~ N(0, N)
                u_k = K_k @ x  # Control input based on the state
                x = A @ x + B @ u_k + v_k  # State update
                trajectory.append(x)
            
            trajectories.append(np.array(trajectory))

        # Plotting the trajectories in 3D
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=figsize)

        # Plot each trajectory
        for i in range(num_trajectories):
            ax.plot(list(range(k_bar+1)), trajectories[i][:, 0], zs=trajectories[i][:, 1], color="gray",alpha=0.5)
        
        for i in range(k_bar+1):
            x,y = plot_ellipse(Sigma_optimal[i])
            ax.plot(np.ones(100)*i, x, zs=y,alpha=0.9)
        ax.set_xlabel(r'$k$',labelpad=20)
        ax.set_ylabel(r'$(x)_1$',labelpad=20)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r'$(x)_2$',labelpad=10, rotation=90)

        ax.view_init(20, 70)
        ax.set_yticks(np.arange(-10, 11, 5))
        ax.set_zticks(np.arange(-10, 11, 5))
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_ylim([-10,10])  
        ax.set_zlim([-10,10])  
        ax.set_xlim([10,0])  
        plt.tight_layout()
        plt.savefig("./Figure11_1{}.pdf".format(label))
        plt.show()

if __name__ == '__main__':
    figure11_1(label="a")
    figure11_1(label="b")
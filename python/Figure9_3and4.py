# Author: Kenji Kashima
# Date  : 2025/09/01
# Note  : pip install control

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
import control
import config

np.random.seed(2)

# Function for the RL4LQR algorithm
def RL4LQR(sigma, iter_Gain, k_update, Npath):
    """
    Simulate RL for LQR

    Parameters
    ----------
    sigma           : scale of exploration noise
    iter_Gain       : number of gain updates
    k_update        : time steps for one gain update (k_update)
    Npath           : number of sample paths

    Returns
    -------
    x_norm_hist_p   : history of state norms
    K_err_hist_p    : history of control gain errors
    Ups_err_hist_p  : history of H errors
    iter_Gain       : number of gain updates
    """

    beta = 0.95  # discount rate
    A = np.array([[0.80, 0.90, 0.86],
                  [0.30, 0.25, 1.00],
                  [0.10, 0.55, 0.50]])  # A-matrix
    B = np.array([[1], [0], [0]])  # B-matrix

    x_dim, u_dim = B.shape  # state & input dimension
    p_dim = (x_dim + u_dim) * (x_dim + u_dim + 1) // 2  + 1 # size of p (integer)

    Q = np.eye(x_dim)  # cost x'Qx
    R = np.eye(u_dim)  # cost u'Ru

    # Optimal gain for comparison
    K_opt, _, _ = control.dlqr(np.sqrt(beta) * A, np.sqrt(beta) * B, Q, R)

    K_ini = np.array([[4.1100, 11.7519, 19.2184]])
    IsUnstable = False
    x_ini = np.zeros((x_dim, 1))  # Initial state of dynamics

    x_norm_hist_p = []
    Ups_err_hist_p = []
    K_err_hist_p = []

    for j in range(Npath):
        
        K_err_hist = []
        Ups_err_hist = []

        p = np.zeros(p_dim)

        K = K_ini
        x = x_ini
        x_hist = x
        for _ in range(iter_Gain):
            # Compute Ups_true for current K for comparison
            Pi = dlyap(np.sqrt(beta) * (A - B @ K).T, np.hstack([np.eye(x_dim), -K.T]) @ np.vstack([np.eye(x_dim), -K]))  # Solve Lyapunov equation (9.22)
            Ups_true = beta * np.hstack([A, B]).T @ Pi @ np.hstack([A, B]) + np.block([[Q, np.zeros((x_dim, u_dim))], [np.zeros((u_dim, x_dim)), R]])   # eq.(9.34)

            # TD learning 

            # initialization of RLS
            p = np.zeros((p_dim,1))     
            Sigma = np.eye(p_dim) * 10

            for k in range(k_update):
               
                K_err_hist.append(np.linalg.norm(K - K_opt) / np.linalg.norm(K_opt))
                Ups_err_hist.append(np.linalg.norm(p_to_Ups(p, x_dim + u_dim) - Ups_true) / np.linalg.norm(Ups_true))

                u = -K @ x + np.random.randn(u_dim, 1) * sigma  # tentative SF + exploration noise
                cost = x.T @ Q @ x + u.T @ R @ u    # stage cost

                phi_pre = phi(x, u)
                x = A @ x  + B @ u                  # One-step simulation
                x_hist= np.hstack((x_hist,x))

                q = phi_pre - beta * phi(x, -K @ x) # line 1 
                denom = 1 + q.T @ Sigma @ q
                p = p + (Sigma @ q @ (cost - q.T @ p)) / denom  # line 2
                Sigma = Sigma - (Sigma @ q @ q.T @ Sigma) / denom   # line 3

            Ups = p_to_Ups(p, x_dim + u_dim)
            K = np.linalg.solve(Ups[x_dim:, x_dim:], Ups[x_dim:, :x_dim])   # eq.(9.24)

            if max(abs(np.linalg.eigvals(A - np.dot(B, K)))) > 1:
                IsUnstable = True
                break

        if not IsUnstable:
            x_norm_hist_p.append(np.linalg.norm(x_hist, axis=0))
            K_err_hist_p.append(K_err_hist)
            Ups_err_hist_p.append(Ups_err_hist)
        else:
            IsUnstable = False
    return (x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain)
    

# Function to compute feature vector phi from x and u; eq.(9.35)
def phi(x, u):
    xu = np.vstack([x, u])              # (n,1)
    H  = xu @ xu.T                      # (n,n)
    phi_vector = np.hstack([H[i, i:] for i in range(H.shape[0])]).reshape(-1, 1)
    return np.concatenate([phi_vector, [[1.0]]])

# Function to convert p vector to Upsilon matrix; eq.(9.35)
def p_to_Ups(p, n):
    p = np.asarray(p).reshape(-1)  # 1D
    p = p[:n*(n+1)//2]                      # drop trailing bias

    H = np.zeros((n, n))
    idx = 0
    for i in range(n):
        H[i, i:] = p[idx:idx + n - i]
        idx += n - i
    return (H + H.T) / 2

# Function to plot results
def plot_results(x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain, label=None):
    grayColor = [0.7, 0.7, 0.7]
    figsize = config.global_config(type= 1)
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    
    # Plot x_norm_hist_p
    ax[0].plot(np.array(x_norm_hist_p).T, color=grayColor)
    ax[0].plot(np.mean(x_norm_hist_p, axis=0), linewidth=2, color='black')
    ax[0].set_xlim([0,len(np.array(x_norm_hist_p).T)-1])
    ax[0].axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Plot Ups_err_hist_p
    ax[1].plot(np.array(Ups_err_hist_p).T, color=grayColor)
    ax[1].plot(np.mean(Ups_err_hist_p, axis=0), linewidth=1.5, color='black')
    ax[1].set_ylabel('Q function error')
    ax[1].set_xlim([0,len(np.array(Ups_err_hist_p).T)-1])
    ax[1].axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.xlabel(r'$k$')
    # Plot K_err_hist_p if iter_Gain > 1
    if iter_Gain > 1:
        ax2 = ax[1].twinx()  # Create a twin y-axis
        ax2.plot(np.mean(K_err_hist_p, axis=0), linewidth=2, color='red')
        ax2.set_ylabel('Gain error', color='red')

    plt.tight_layout()
    
    if label is not None:
        plt.savefig("./Figure9_{}.pdf".format(label))
    plt.show()

if __name__ == '__main__':
    # Running the RL4LQR for different scenarios
    k_update = 50
    Npath = 20

    # Figure 9.3(a)
    iter_Gain = 1
    sigma = 2
    x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain = RL4LQR(sigma, iter_Gain, k_update, Npath)
    plot_results(x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain,label="3a")
    
    # Figure 9.3(b)
    sigma = 10
    x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain = RL4LQR(sigma, iter_Gain, k_update, Npath)
    plot_results(x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain,label="3b")

    # Figure 9.4
    iter_Gain = 5
    x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain = RL4LQR(sigma, iter_Gain, k_update, Npath)
    plot_results(x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain, label="4")


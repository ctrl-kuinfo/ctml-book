# Author: Kenji Kashima
# Date  : 2025/06/21
# Note  : pip install control

import numpy as np
import matplotlib.pyplot as plt
import control
import config

np.random.seed(1)  # Random seed
x_max = 1/2

def lqr_control(A, B_u, Q, R, Qf, k_bar):
    """
    Finite-horizon discrete-time LQR controller
    Calculates feedback gains K and cost-to-go matrices Sigma for k=0...k_bar

    Parameters
    ----------
    A, B_u : system matrices (x_{k+1} = A x_k + B_u u_k)
    Q, R   : stage cost matrices
    Qf     : terminal cost matrix
    k_bar  : horizon length (number of steps)

    Returns
    -------
    K     : list of feedback gain matrices, length k_bar
    Sigma : list of cost-to-go matrices, length k_bar+1
    """
    # Initialize lists to store Sigma_k and K_k
    Sigma = [None] * (k_bar + 1)  # Sigma[k] = cost-to-go at step k
    K = [None] * k_bar            # K[k]    = feedback gain at step k

    # Terminal cost
    Sigma[k_bar] = Qf

    # Backward Riccati recursion
    for k in range(k_bar - 1, -1, -1):
        # Compute R̃_k = R + B_u^T Sigma[k+1] B_u
        R_tilde = R + B_u.T @ Sigma[k + 1] @ B_u
        # Compute S̃_k = A^T Sigma[k+1] B_u
        S_tilde = A.T @ Sigma[k + 1] @ B_u
        # Compute Q̃_k = Q + A^T Sigma[k+1] A
        Q_tilde = Q + A.T @ Sigma[k + 1] @ A

        # Feedback gain K[k] = R̃_k^{-1} S̃_k^T
        K[k] = np.linalg.solve(R_tilde, S_tilde.T)

        # Update cost-to-go Sigma[k]
        Sigma[k] = Q_tilde - S_tilde @ np.linalg.solve(R_tilde, S_tilde.T)

    return K, Sigma


def simulate_lq_control(A, B_u, B_v, C, mu, Sigma, K, k_bar, x0, mode, Rw=1e-4, Rv=1.0, v=None, w=None):
    """
    Simulate LQG controller using Algorithm 1 order
    Records true state, state estimates, inputs, measurements, and covariance

    Parameters
    ----------
    A, B_u, B_v, C : system matrices
    mu             : Initial mean estimate
    Sigma          : Initial covariance
    K              : list of finite-horizon LQR gains K[k]
    k_bar          : horizon length
    x0             : initial state vector
    mode           : 'lqr', 'lqg_kalman', 'lqg_pred', or 'lqr_m'
    Rw, Rv         : measurement and process noise covariances
    v              : optional pre-generated process noise sequence
    w              : optional pre-generated observation noise sequence

    Returns
    -------
    x_true : true state trajectories (x_dim x (k_bar+1))
    x_hat  : state estimates        (x_dim x (k_bar+1))
    u      : control inputs         (k_bar,)
    y      : measurements           (k_bar,)
    Sigmas : one-step prediction covariance matrices    (x_dim x x_dim x (k_bar+1))
    x_check: corrected state estimates        (x_dim x (k_bar))
    Sigmac : corrected covariance matrices    (x_dim x x_dim x (k_bar))
    """
    x_dim = A.shape[0]

    # Generate noise sequences if not provided
    if v is None:
        v = np.sqrt(Rv) * np.random.randn(k_bar)  # process noise v_k ~ N(0, Rv)
    if w is None:
        w = np.sqrt(Rw) * np.random.randn(k_bar)  # observation noise w_k ~ N(0, Rw)

    # Allocate containers
    x_true  = np.zeros((x_dim, k_bar + 1))
    x_hat   = np.zeros((x_dim, k_bar + 1))
    Sigmas  = np.zeros((x_dim, x_dim, k_bar + 1))
    x_check = np.zeros((x_dim, k_bar))
    Sigmac  = np.zeros((x_dim, x_dim, k_bar))
    u       = np.zeros(k_bar)
    y       = np.zeros(k_bar)

    # Initial conditions
    x_true[:, 0] = x0
    x_hat[:, 0]  = mu
    Sigmas[:, :, 0] = Sigma

    # Main loop over time steps
    for k in range(k_bar):
        # If perfect state measurement (LQR)
        if mode == 'lqr':
            u[k] = (-K[k] @ x_true[:, k]).item()
        else:
            # For prediction mode, use prior estimate
            if mode == 'lqg_pred':
                u[k] = (-K[k] @ x_hat[:, k]).item()

            # 1) Receive measurement y_k
            y[k] = (C @ x_true[:, k] + w[k]).item()

            # 2) Compute Kalman gain components: M̃_k, Ľ_k, Ȟ_k
            M_tilde  = C @ Sigma @ C.T + Rw      # innovation covariance
            L_check  = Sigma @ C.T               # cross-covariance
            H_check  = np.linalg.solve(M_tilde, L_check.T).T # Kalman gain

            # 3) Posterior update of state estimate x̌_k
            innov    = y[k] - (C @ x_hat[:, k]).item()
            x_check[:, k]  = x_hat[:, k] + H_check.flatten() * innov

            # 4) Posterior covariance Σ̌_k
            Sigma_check = Sigma - H_check @ L_check.T
            Sigmac[:, :, k] = Sigma_check

            # Choose input using updated estimate for LQG
            if mode == 'lqg_kalman':
                u[k] = (-K[k] @ x_check[:, k]).item()

            # 5) Time update (prior for next step)
            x_hat[:, k+1] = (A @ x_check[:,k].reshape(x_dim,1) + B_u * u[k]).flatten()
            Sigma = A @ Sigma_check @ A.T + Rv * (B_v @ B_v.T)

        # 6) Propagate true system
        x_true[:, k+1] = (A @ x_true[:, k].reshape(x_dim,1) + B_u * u[k] + B_v * v[k]).flatten()
        Sigmas[:, :, k+1] = Sigma
        
    return x_true, x_hat, u, y, Sigmas, x_check, Sigmac


def figure5_3a(x_true, x_true_k, x_LQRmm, x_LQRm, k_bar):
    figsize = config.global_config(type=0)
    t = np.arange(k_bar + 1)
    plt.figure(figsize = figsize)
    plt.plot(t, x_LQRm[0], 'k--', label='LQR (white noise)')
    plt.plot(t, x_LQRmm[0], 'k', label='LQR')
    plt.plot(t, x_true[0], 'b', label='LQG')
    plt.plot(t, x_true_k[0], 'r', label='LQG Kalman')

    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, k_bar])
    plt.ylim([-x_max, x_max])
    plt.savefig("./Figure5_3a.pdf")
    plt.show()

def figure5_3b(x_true, x_hat, y, Sigmas, x_check, Sigmac, k_bar):
    figsize = config.global_config(type=0)
    t = np.arange(k_bar + 1)
    t_c = np.arange(k_bar)
    plt.figure(figsize=figsize)
    plt.plot(t, x_true[0], 'b', label='Position')
    plt.plot(t_c, y, 'r', label='Measurements')
    plt.plot(t, x_hat[0], 'b--', label='Estimate')
    sd = np.sqrt(Sigmas[0, 0, :])
    plt.fill_between(t, x_hat[0] - sd, x_hat[0] + sd, color='blue', alpha=0.2)

    # Plot corrected estimate
    plt.xlabel('$k$')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, 100])
    plt.ylim([-0.2, 0.2])
    plt.savefig("./Figure5_3b.pdf")
    plt.show()

def figure5_3c(x_true, x_hat, Sigmas, x_check, Sigmac, k_bar):
    figsize = config.global_config(type=0)
    t = np.arange(k_bar + 1)
    plt.figure(figsize=figsize)
    plt.plot(t, x_true[1], 'b', label='Velocity')
    plt.plot(t, x_hat[1], 'b--', label='Estimate')
    sd = np.sqrt(Sigmas[1, 1, :])
    plt.fill_between(t, x_hat[1] - sd, x_hat[1] + sd, color='blue', alpha=0.2)

    # Plot corrected estimate
    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, 100])
    plt.ylim([-x_max, x_max])
    plt.savefig("./Figure5_3c.pdf")
    plt.show()

def figure5_3d(x_true, x_hat, Sigmas, x_check, Sigmac, k_bar):
    figsize = config.global_config(type=0)
    t = np.arange(k_bar + 1)
    plt.figure(figsize=figsize)
    plt.plot(t, x_true[2], 'b', label='Colored noise')
    plt.plot(t, x_hat[2], 'b--', label='Estimate')
    sd = np.sqrt(Sigmas[2, 2, :])
    plt.fill_between(t, x_hat[2] - sd, x_hat[2] + sd, color='blue', alpha=0.2)

    # Plot corrected estimate
    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, 100])
    x_max = 2
    plt.ylim([-x_max, x_max])
    plt.savefig("./Figure5_3d.pdf")
    plt.show()


def create_double_integrator(Ts):
    """
    Create discrete-time double integrator P = 1/s^2
    State: [position, velocity]
    """
    A_c = np.array([[0, 1], [0, 0]])
    B_c = np.array([[0], [1]])
    C_c = np.array([[1, 0]])
    P_c = control.StateSpace(A_c, B_c, C_c, 0)
    P_d = control.c2d(P_c, Ts, method='zoh')
    return P_d.A, P_d.B, P_d.C

def create_noise_model(Ts):
    # --- 1. Noise Model Normalization ---
    # Continuous-time noise model F = 1/(s+0.3)
    s = control.TransferFunction.s
    F_c = 1 / (s + 0.3)
    F_ss = control.ss(F_c)
    F_d = control.c2d(F_ss, Ts, method='tustin')
    Aw, Bw, Cw = F_d.A, F_d.B, F_d.C

    # --- 2. Noise Model Normalization ---
    # Solve discrete Lyapunov equation: P = A * P * A^T + B * B^T
    Pw = control.dlyap(Aw, Bw @ Bw.T)
    variance_amplification = (Cw @ Pw @ Cw.T)[0, 0]  # Scalar variance
    Bw = Bw / np.sqrt(variance_amplification)

    return Aw, Bw, Cw


# --- 1. System and Noise Model Setup ---
Ts = 0.1
Aw, Bw, Cw = create_noise_model(Ts)
nw = Aw.shape[0]
A_r, B_r, C_r = create_double_integrator(Ts)
nr = A_r.shape[0]

# --- 2. Augmented State-Space Model Construction ---
A = np.block([
    [A_r,                 B_r @ Cw],
    [np.zeros((nw, nr)),  Aw]
]) 
B_u = np.vstack([B_r, np.zeros((nw,1)) ])  # Control input matrix
B_v = np.vstack([np.zeros((nr,1)), Bw])    # Disturbance input matrix (noise)
C = np.block([ C_r, np.zeros((1,nw)) ])    # Output matrix
x_dim = A.shape[0]

# --- 3. LQR and Noise Parameter Definition ---
# LQR parameters
Q = np.diag([1, 10] + list(np.zeros(nw)))   # State cost matrix
R = 1e-4                                     # Control input cost
Qf = Q                                       # Final state cost
k_bar = 300                                  # Total time steps

# Noise properties
Rv = 1                                       # Process noise covariance (v_k ~ N(0, 1))
Rw = 1e-4                                    # Measurement noise covariance (w_k ~ N(0, 1e-4))

# --- 4. Initial State and Noise Sequence Generation ---
# process noise v_k ~ N(0, Rv) 
process_noise = np.sqrt(Rv) * np.random.randn(k_bar)          
# observation noise w_k ~ N(0, Rw) 
obs_noise = np.sqrt(Rw) * np.random.randn(k_bar)

# Initial state
mu = np.zeros(x_dim)    # initial mean
Sigma = np.eye(x_dim)   # initial covariance
x0 = np.random.multivariate_normal( mu, Sigma )*0.1   # Initial state x0 ~ N(mu, Sigma)

# --- 5. LQR Gain Calculation ---
# Compute LQR feedback gains
K, P = lqr_control(A, B_u, Q, R, Qf, k_bar)

# Compute LQR feedback gains assuming white noise
K_lqr, P_lqr = lqr_control(A_r, B_r, Q[0:2,0:2], R, Qf[0:2,0:2], k_bar)

# --- 6. Simulation Execution ---
# Simulate LQR (no Σ needed) and LQG controllers

# LQG optimal w/ one-step prediction (colored noise)
x_true, x_hat_LQG, u, y, Sigmas, x_check, Sigmac = simulate_lq_control(A, B_u, B_v, C, mu, Sigma, K, k_bar, x0, mode='lqg_pred', Rw=Rw, Rv=Rv, v = process_noise, w = obs_noise)

# LQG optimal w/ Kalman filtering (colored noise)
x_true_k, x_hat_LQG_k, u_k, y_k, Sigmas_k, x_check_k, Sigmac_k = simulate_lq_control(A, B_u, B_v, C, mu, Sigma, K, k_bar, x0, mode='lqg_kalman', Rw=Rw, Rv=Rv, v = process_noise, w = obs_noise)

# LQR optimal (white noise)
x_LQRm, _, u_LQRm, _, _, _, _  = simulate_lq_control(A_r, B_r, B_r, C_r, mu[0:2], Sigma[0:2,0:2], K_lqr, k_bar, x0[0:2], mode='lqr', Rw=Rw, Rv=Rv, v = process_noise, w = obs_noise)

# LQR optimal (colored noise)
K_lqr_aug = [np.pad(Ki, ((0,0),(0, x_dim - Ki.shape[1])), mode='constant') for Ki in K_lqr]    # Pad K to match dimensions
x_LQRmm, _, u_LQRmm, _, _, _, _  = simulate_lq_control(A, B_u, B_v, C, mu, Sigma, K_lqr_aug, k_bar, x0, mode='lqr', Rw=Rw, Rv=Rv, v = process_noise, w = obs_noise)


if __name__ == '__main__':
    figure5_3a(x_true, x_true_k, x_LQRmm, x_LQRm, k_bar)
    figure5_3b(x_true, x_hat_LQG, y, Sigmas, x_check, Sigmac, k_bar)
    figure5_3c(x_true, x_hat_LQG, Sigmas, x_check, Sigmac, k_bar)
    figure5_3d(x_true, x_hat_LQG, Sigmas, x_check, Sigmac, k_bar)
    
    # print('performance matched LQR')
    # print(x_LQRm[0].T @ x_LQRm[0] + u_LQRm.T @ u_LQRm*R)
    # print('performance mismatched LQR')
    # print(x_LQRmm[0].T @ x_LQRmm[0] + u_LQRmm.T @ u_LQRmm*R)
    # print('performance LQG')
    # print(x_true[0].T @ x_true[0] + u.T @ u*R)
    # print('performance LQG Kalman')
    # print(x_true_k[0].T @ x_true_k[0] + u_k.T @ u_k*R)
# Author: Kenji Kashima
# Date  : 2025/09/01
# Note  : pip install control

import control
import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(1)

def create_noise_model(Ts):
    # --- 1. Noise Model time discretization ---
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

def figure3_4a(Aw, Bw, Cw):
    figsize = config.global_config(type=1)
    dsys = control.ss(Aw, Bw, Cw, 0, dt=True)
    mag, _, omega = control.frequency_response(dsys, np.arange(0.001, np.pi, 0.001))
    plt.figure(figsize=figsize)
    plt.semilogx(omega, mag, linewidth=1.0, color='blue', label='frequency weight')
    plt.plot([0.001,0.01,0.6,np.pi], [8,8,0,0], linewidth=2, linestyle='--', color='red', label='prior information')
    plt.xlabel(r'$\varpi$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Figure3_4a.pdf")
    plt.show()

def figure3_4b( Aw, Bw, Cw, k_bar ):
    x_bar_dim = Aw.shape[0]
    v = np.random.randn(k_bar+1) # white noise
    x_bar = np.zeros((x_bar_dim, k_bar + 1)) # state of weighting filter
    y = np.zeros(k_bar + 1) # colored noise
    
    for k in range(k_bar):
        x_bar[:, k+1] = Aw @ x_bar[:, k] + Bw.ravel() * v[k]
        y[k+1] = (Cw @ x_bar[:, k+1])[0]

    figsize = config.global_config(type=1)
    plt.figure(figsize=figsize)
    plt.xlabel(r'$k$')
    plt.xlim(0,200)
    plt.ylim(-2,2)
    plt.stairs(v[:201], label='white')
    plt.stairs(y[:201], linewidth=1.0, label='colored')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure3_4b.pdf")
    plt.show()

# Parameters for noise model
Ts = 0.1
Aw, Bw, Cw = create_noise_model(Ts)
# nw = Aw.shape[0]

if __name__ == '__main__':
    figure3_4a(Aw, Bw, Cw)
    figure3_4b(Aw, Bw, Cw, k_bar=200)
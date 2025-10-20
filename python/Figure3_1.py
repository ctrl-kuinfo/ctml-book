# Author: Kenji Kashima
# Date  : 2025/04/01

import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(100)

def figure3_1a(k_bar=9):
    '''
        k_bar - number of time steps
    '''
    figsize = config.global_config(type=1)

    n_tmp = 1000
    n_x = 2 * n_tmp + 1   # number of discrete x points over [-1, 1]
    x = np.linspace(-1, 1, n_x)
    dx = x[1] - x[0]      # discretization interval; i'th cell: [ x[i]-dx/2, x[i]+dx/2 ]

    # Build transition probability matrix P (approximate uniform noise over interval)
    P = np.zeros([n_x, n_x])
    for i in range(n_x):
        # fb(x) = x + 0.1*(x - x^3)  
        # noise amplitude: c(x) = 0.5*(1 - |x|)  
        upper = x[i] + 0.1 * (x[i] - x[i]**3) + 0.5 * (1 - np.abs(x[i]))
        lower = x[i] + 0.1 * (x[i] - x[i]**3) - 0.5 * (1 - np.abs(x[i]))
        idx_u = int(np.floor(upper * n_tmp + 0.5)) + n_tmp
        idx_l = int(np.ceil(lower * n_tmp - 0.5)) + n_tmp
        for j in range(idx_l, idx_u + 1):
            P[j, i] += 1.0 / (idx_u - idx_l + 1)

    # Define initial distribution φₓ₀ = 1 on [-0.5, 0.5]
    init = np.where((x >= -0.5) & (x <= 0.5), 1.0, 0.0)
    # Normalize so that sum(init)*dx = 1
    init /= (np.sum(init) * dx)

    # Initialize phi array: each column is the state distribution at time k
    phi = np.zeros([n_x, k_bar + 1])
    phi[:, 0] = init

    # Propagate the distribution by multiplying with P at each time step
    for k in range(k_bar):
        phi[:, k + 1] = P @ phi[:, k]
        # If needed, re-normalize to guard against numerical drift:
        # phi[:, k+1] /= (np.sum(phi[:, k+1]) * dx)

    # Plot from k=0 onward
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
    for k in range(k_bar + 1):
        ax.plot3D(np.full(n_x, k), x, phi[:, k])

    ax.set_xlabel(r'$k$', labelpad=20)
    ax.set_ylabel(r'${\rm x}$', labelpad=20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\varphi_{x_k}$', rotation=90, labelpad=20)
    ax.view_init(26, -107)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure3_1a.pdf")
    plt.show()


def figure3_1b(k_bar: int = 50, n_sample: int = 10):
    '''
        k_bar       - total number of steps
        n_sample  - number of sample trajectories
    '''
    config.global_config()

    plt.figure(figsize=(8, 7))
    for _ in range(n_sample):
        x = np.zeros(k_bar)
        x[0] = np.random.rand() - 0.5  # initial state in [-0.5, 0.5]
        for k in range(k_bar - 1):
            noise = np.random.rand() - 0.5  # uniform in [-0.5, 0.5]
            x[k + 1] = x[k] + 0.1 * (x[k] - x[k]**3) + noise * (1 - np.abs(x[k]))
        plt.plot(x, linewidth=2)

    plt.xlim([0, k_bar - 1])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.grid()
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("./Figure3_1b.pdf")
    plt.show()


if __name__ == '__main__':
    figure3_1a(k_bar=9)
    figure3_1b(k_bar=30, n_sample=10)

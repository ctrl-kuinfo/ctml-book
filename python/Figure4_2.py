# Author: Kenji Kashima
# Date  : 2025/04/01
# Note  : pip install control

import control as ctl
import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(1)

def figure4_2(k_bar:int = 800, T_c:float=0.01):
    '''
        k_bar - total steps
        T_c - discrete-time stepsize [s]
    '''
    figsize = config.global_config(type=2)

    # parameters for a continuous-time system
    Ac = np.array([[0,4],[-3,2]])
    Bc = np.array([[0],[1]])
    Cc = np.array([[-0.3, -4]])
    csys = ctl.ss(Ac,Bc,Cc,0) # build CT system

    dsys = ctl.c2d(csys, T_c)         # build DT system from the CT system
    [A ,B ,C ,D ] = ctl.ssdata(dsys)  # get parameters of the DT system
                                      # Note: D=0
    u_max = 3.2    # view range
    d = 1.0        # quantization unit
    x0 = np.array([[1],[0.5]])  # init state

    fig = plt.figure(figsize=figsize)
    # u_k = y_k
    y = np.zeros(k_bar)
    x_k = x0
    for k in range(k_bar):
        y[k] = (C @ x_k).item()
        u_k = y[k]
        x_k = A @ x_k + B * u_k

    ax = fig.subplots(1,3)
    ax[0].stairs(y,label=r'$y_k$',linewidth=1.0, color = 'black',zorder=10)

    ax[0].set_title(r'$u_k=y_k$')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel(r'$k$')
    ax[0].tick_params()
    ax[0].axis([0,k_bar,-u_max,u_max])

    # u_k = Q(y_k)
    y = np.zeros(k_bar)
    u = np.zeros(k_bar)
    x_k = x0
    for k in range(k_bar):
        y[k] = (C @ x_k).item()
        u[k] = np.floor((y[k] + d/2) / d) * d   # nearest grid to y[k] 
        x_k = A @ x_k + B * u[k]
    ax[1].stairs(u,label=r'$u_k$',linewidth=1.0, color = [0.7,0.7,0.7],zorder=10)
    ax[1].stairs(y,label=r'$y_k$',linewidth=1.0, color = 'black',zorder=10)

    ax[1].set_title(r'$u_k=\mathcal{Q}(y_k)$')
    ax[1].legend()
    ax[1].grid(zorder=0)
    ax[1].set_xlabel(r'$k$')
    ax[1].axis([0,k_bar,-u_max,u_max])
    
    # u_k = Q(y_k + z_k)
    y = np.zeros(k_bar)
    u = np.zeros(k_bar)
    x_k = x0
    for k in range(k_bar):
        y[k] = (C @ x_k).item()
        z_k = np.random.rand() - 0.5 # uniform distribution [-0.5,0.5]
        u[k] = np.floor((y[k] + z_k + d/2) / d) * d
        x_k = A @ x_k + B * u[k]
    ax[2].stairs(u,label=r'$u_k$',linewidth=1.0, color = [0.7,0.7,0.7],zorder=10)
    ax[2].stairs(y,label=r'$y_k$',linewidth=1.0, color = 'black',zorder=10)

    ax[2].set_title(r'$u_k=\mathcal{Q}(y_k+z_k)$')
    ax[2].legend()
    ax[2].grid(zorder=0)
    ax[2].set_xlabel(r'$k$')
    ax[2].tick_params()
    ax[2].axis([0,k_bar,-u_max,u_max])

    plt.tight_layout()
    plt.savefig("./Figure4_2.pdf")
    plt.show()
    
if __name__ == '__main__':
    figure4_2()
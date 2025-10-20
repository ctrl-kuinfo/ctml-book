# Author: Kenji Kashima
# Date  : 2025/09/01
import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(23)

def figure11_4(N_k= 100,label="a"):
    """
    Runs a SGD simulation for mean estimation.

    Parameters:
        N_k : simulation time length.
    """
    figsize = config.global_config(type= 1)
    C = [0.6, 0.6, 1.0, 1.0]        # constants for step size
    alpha = [1.0, 0.3, 1.0, 1.5]    # decay rate for step size
    N_setting = len(C)              # Number of settings

    p_ini = 1                       # initial value for SGD
    p_list = np.zeros((N_setting,N_k+1))
    y_list = np.zeros((N_setting,N_k))
    plt.figure(figsize=figsize)  
    p_list[:,0] = np.ones_like(p_list[:,0]) * p_ini 
    for setting in range( N_setting ):
        for k in range(N_k):        
            p = p_list[setting,k]
            z = np.random.randn()   # sampling of z_k 
            y_list[setting,k] = p - z   # Fig 11.4   
            # y_list[setting,i] = ( p > z ) - 0.5    # 演習 11.5 
            p_list[setting,k+1] = p - C[setting]/((k+1)**alpha[setting]) * y_list[setting,k]
        plt.plot(p_list[setting,:],label=r"$C={}, \alpha={}$".format(C[setting],alpha[setting]))

    plt.scatter(0,p_ini,marker='o',s=80,clip_on=False,color='black',label= "Initial Value",zorder=20)
    plt.ylabel(r"$p_k$")
    plt.xlabel(r"$k$")
    plt.xlim([0,N_k])
    plt.ylim([-2,2])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure11_4{}.pdf".format(label))
    plt.show()

if __name__ == '__main__':
    figure11_4(N_k=100,label="a")   
    figure11_4(N_k=5000,label="b") 

# Author: Kenji Kashima
# Date  : 2025/04/01
# Note  : pip install control

import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(100)

a = 1.1;        # system matrix
alpha = -0.2;   # parameter in control law
x_max = 5;      # view range

def figure4_1a(k_bar:int=50):
    '''
        k_bar - total steps
    '''
    figsize = config.global_config(type=1)
    
    x = np.ones(k_bar+1)       # state with v=0
    for k in range(k_bar):
        u = alpha * (a + alpha) ** k  # control law
        x[k+1] = a * x[k] + u
    
    plt.figure(figsize=figsize)
    plt.plot(x,linewidth=1,color=[0,0,0])
    plt.xlim([0,k_bar])
    plt.ylim([-x_max,x_max])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.tight_layout()
    plt.grid()
    plt.savefig("./Figure4_1a.pdf")
    plt.show()

def figure4_1b(k_bar:int=50,n_sample:int=20):
    '''
        k_bar - total steps
        n_sample - number of samples
    '''
    figsize = config.global_config(type=1)

    x = np.ones(k_bar+1)       # state with v=0
    x_v = np.ones([n_sample,k_bar+1]) # state with v~N(0,1)
    for k in range(k_bar):
        v = np.random.randn(n_sample) # n_sample independent standard normal distributions
        u = alpha * (a + alpha) ** k  # control law
        x[k+1] = a * x[k] + u
        x_v[:,k+1] = a * x_v[:,k] + u + 0.1 * v
    
    plt.figure(figsize=figsize)
    for i in range(n_sample):
        plt.plot(x_v[i,:],linewidth=0.2,color=[0.7,0.7,0.7])

    plt.plot(x,linewidth=1,color=[0,0,0])
    plt.xlim([0,k_bar])
    plt.ylim([-x_max,x_max])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.tight_layout()
    plt.grid()
    plt.savefig("./Figure4_1b.pdf")
    plt.show()

if __name__ == '__main__':
    figure4_1a(k_bar=50)
    figure4_1b(k_bar=50,n_sample=20)

# Author: Kenji Kashima
# Date  : 2025/04/01

import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(100)

x_max = 10    # range of y_lim

def figure3_3a(k_bar:int=50,n_sample:int=20,a:float=0.5):
    '''
        k_bar - total steps
        n_sample - number of samples
        a - gain
    '''
    config.global_config()
    

    x = np.zeros([n_sample,k_bar+1])
    x[:,0] = np.random.randn(n_sample).T

    step = np.random.randn(n_sample).T
    for k in range(k_bar):
        x[:,k+1] =  a*x[:,k] + step
    
    plt.figure(figsize=(8,7))
    for x_sample in x:
        plt.plot(np.arange(0,k_bar+1),x_sample, linewidth=2,alpha=0.8)

    plt.xlabel(r'$k$',)
    plt.ylabel(r'$x_k$')
    plt.ylim([-x_max,x_max])
    plt.xlim([0,k_bar])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure3_3a.pdf")
    plt.show() 


def figure3_3b(k_bar:int=50,n_sample:int=20,a:float=0.5):
    '''
        k_bar - total steps
        n_sample - number of samples
        a - gain
    '''
    figsize = config.global_config(type=1)

    x = np.zeros([n_sample,k_bar+1])
    x[:,0] = np.random.randn(n_sample).T

    for k in range(k_bar):
        noise = np.random.randn(n_sample).T
        x[:,k+1] =  a*x[:,k] + noise*np.sqrt(3)
    
    plt.figure(figsize=figsize)
    for x_sample in x:
        plt.plot(np.arange(0,k_bar+1),x_sample, linewidth=2,alpha=0.8)

    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.ylim([-x_max,x_max])
    plt.xlim([0,k_bar])
    plt.grid()    
    plt.tight_layout()
    plt.savefig("./Figure3_3b.pdf")
    plt.show() 

if __name__ == '__main__':
    figure3_3a(k_bar=10,n_sample=20)
    figure3_3b(k_bar=10,n_sample=20)
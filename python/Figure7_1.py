# Author: Kenji Kashima
# Date  : 2025/09/11

import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(24)

n_x = 100;          # n_x - number of x-grid for plot
x_p = np.linspace(0,1,n_x)  # x grid points for plot

def phi(x:float)->np.ndarray:
    '''
        Feature mapping
        x - input
    '''
    return  np.array([1, x, x**2, x**3, x**4, 
                      x**5, x**6, x**7, x**8, x**9 ])

n_f = phi(0).size   # number of features
Phi_p = np.column_stack([phi(xi) for xi in x_p])    # matrix phi for plot

def f_true(x:float)->float:
    '''
        x - input
    '''
    return 2*np.sin(5*x)

def figure7_1a(n_sample:int=20):
    '''
        n_sample - number of sample functions
    '''
    figsize = config.global_config(type=1)
    plt.figure(figsize=figsize)
    plt.plot(x_p, np.zeros_like(x_p), linewidth=3)

    for _ in range (n_sample):
        para = np.random.randn(n_f)      
        fx =  para @ Phi_p                 
        plt.plot(x_p,fx,linewidth= 0.5, color=[0.7,0.7,0.7])    
    
    plt.xlabel(r'$\rm x$')
    plt.ylabel(r'$f({\rm x})$')
    plt.xlim([0,1])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure7_1a.pdf")
    plt.show()


def figure7_1b(sigma_sq, n_sample:int=20, s_bar:int=8):
    '''
        sigma_sq - list of squared SD
        n_sample - number of sample functions
        s_bar - number of data
    '''
    
    x = np.linspace(0,1,s_bar)     # x for data
    Phi = np.column_stack([phi(x_s) for x_s in x])    # matrix phi in (7.10)
    y = np.array([f_true(x_s) + np.random.randn() for x_s in x])  # noisy observation

    size_sigma_sq = len(sigma_sq)
    mean = np.zeros([n_f,size_sigma_sq])
    cov = np.zeros([n_f,size_sigma_sq*n_f])

    for l in range(size_sigma_sq):
        tmp = np.eye(n_f) - Phi @ np.linalg.inv(Phi.T @ Phi + sigma_sq[l] * np.eye(s_bar)) @ Phi.T
        cov[:,n_f*l:n_f*(l+1)] = (tmp+tmp.T)/2
        mean[:,l]  = Phi @ np.linalg.inv(Phi.T @ Phi + sigma_sq[l] * np.eye(s_bar)) @ y.T

    figsize = config.global_config(type=1)
    plt.figure(figsize=figsize)

    plt.plot(x_p, (mean[:,0] @ Phi_p).flatten(), linewidth= 2,label=r'$\sigma=0.5$',zorder=10)
    plt.plot(x_p, (mean[:,1] @ Phi_p).flatten(), linewidth= 2,label=r'$\sigma=10$')
    plt.plot(x_p, (mean[:,2] @ Phi_p).flatten(), linewidth= 2,label=r'$\sigma=10^{-3}$')

    # Sampling from the posterior
    for _ in range (n_sample):
        para = np.random.multivariate_normal(mean[:,0],cov[:,0:n_f])      
        fx =  para @ Phi_p                   
        plt.plot(x_p,fx,linewidth= 0.3,color=[0.7,0.7,0.7])

    plt.scatter(x,y,marker='o',label=r'${\rm y}_s$')
    plt.legend()
    
    plt.xlabel(r'$\rm x$')
    plt.ylabel(r'$f({\rm x})$')
    plt.xlim([0,1])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure7_1b.pdf")
    plt.show()

if __name__ == '__main__':
    figure7_1a(n_sample=20)

    sigma_sq = [0.5, 100, 10**(-6) ]  # standard deviation sigma = 0.5, 10, 10^{-3}

    figure7_1b(sigma_sq, n_sample=20, s_bar=20)

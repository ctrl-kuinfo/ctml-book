# Author: Kenji Kashima
# Date  : 2025/09/11
# Note  : pip install cvxpy
#  (important!!!) You need to install cvxpy first!!!

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import Minimize,Variable,Problem
from cvxpy.atoms import norm
from matplotlib.ticker import MultipleLocator
import config

np.random.seed(5)

n_x = 100;          # n_x - number of x-grid for plot
x_p = np.linspace(0,1,n_x)  # x grid points for plot

def phi(x:float)->np.ndarray:
    '''
        x - input
    '''
    return  np.array([1, x, x**2, x**3, x**4, 
                      x**5, x**6, x**7, x**8, x**9])

n_f = phi(0).size   # number of features
Phi_p = np.column_stack([phi(xi) for xi in x_p])    # matrix phi for plot

def f_true(x:float)->float:
    '''
        x - input
    '''
    return 2*np.sin(5*x)

f_real = np.array([f_true(x) for x in x_p])

def figure7_4(sigma_sq = 0.01, s_bar:int=30):
    '''
        sigma_sq - squared SD
        s_bar - number of data points
    '''

    # generate data from U(0,1)
    x = np.linspace(0,1,s_bar)     # x for data
    Phi = np.column_stack([phi(x_s) for x_s in x])    # matrix phi in (7.10)
    y = np.array([f_true(x_s) + np.random.randn() for x_s in x])  # noisy observation


    # optimization 1 Naive

    para = Variable(n_f)
    obj = Minimize(norm(para @ Phi - y))
    prob = Problem(obj)
    prob.solve()
    para_naive = para.value

    # optimization 2 Lasso
    para = Variable(n_f)
    obj = Minimize(norm(para @ Phi - y) ** 2 + sigma_sq * norm(para,1) )
    prob = Problem(obj)
    prob.solve()
    para_lasso = para.value

    # optimization 3 Ridge
    para = Variable(n_f)
    obj = Minimize(norm(para @ Phi - y) ** 2 + sigma_sq * norm(para) ** 2)
    prob = Problem(obj)
    prob.solve()
    para_ridge = para.value

    NAIVE_dat = para_naive @ Phi_p
    LASSO_dat = para_lasso @ Phi_p
    RIDGE_dat = para_ridge @ Phi_p

    # Figure 7.4(a)
    figsize = config.global_config(type= 1)
    plt.figure(figsize=figsize)
    plt.scatter(x,y,marker='o',label=r'${\rm y}_s$')
    plt.plot(x_p,NAIVE_dat,'k',linewidth=2,label='Least Square')
    plt.plot(x_p,RIDGE_dat,'b',linewidth=2,label='Ridge')
    plt.plot(x_p,LASSO_dat,'r',linewidth=2,label='Lasso')
    plt.plot(x_p,f_real,"-.",linewidth=2,label=r'$2\sin(5{\rm x}_s)$')
    plt.axis([0,1,-5,5])
    plt.xlabel(r'$\rm x$')
    plt.ylabel(r'$f({\rm x})$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure7_4a.pdf")
    
    
    # Figure 7.4(b)
    plt.figure(figsize=figsize)
    plt.scatter(np.arange(1,n_f+1),np.abs(para_ridge),marker='x',s=60,clip_on=False,label='Ridge')
    plt.scatter(np.arange(1,n_f+1),np.abs(para_lasso),marker='o',s=60,clip_on=False,label='Lasso')
    plt.xlabel(r'$i$')
    # plt.ylabel(r'$i$-th coefficient')
    plt.legend()
    plt.xlim([1,n_f])
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.tight_layout()
    plt.grid()
    plt.savefig("./Figure7_4b.pdf")

    plt.show()

if __name__ == '__main__':
    # weight of regularization (= observation noise covariance) 
    sigma_sq = 0.01

    figure7_4(sigma_sq, s_bar=30)
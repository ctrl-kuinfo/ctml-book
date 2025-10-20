# Author: Kenji Kashima
# Date  : 2025/09/01
# pip install sympy

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import config

np.random.seed(33)

# 関数 L, L1, L2 の定義（記号式）
p = sp.Symbol('p')

L1_sym = p**2 / 2 + p / 2
L2_sym = - p * sp.cos(10 * p) / 20 + sp.sin(10 * p) / 200 - p / 2
L_sym = ( L1_sym + L2_sym ) / 2 

grad_L_sym  = sp.diff(L_sym, p)
grad_L1_sym = sp.diff(L1_sym, p)
grad_L2_sym = sp.diff(L2_sym, p)
print('grad_L1_sym = ' , grad_L1_sym)
print('grad_L2_sym = ' , grad_L2_sym)

# 数値計算用に lambdify
L        = sp.lambdify(p, L_sym, 'numpy')
grad_L   = sp.lambdify(p, grad_L_sym, 'numpy')
grad_L1  = sp.lambdify(p, grad_L1_sym, 'numpy')
grad_L2  = sp.lambdify(p, grad_L2_sym, 'numpy')

def figure11_5a():
    figsize = config.global_config(type= 1)
    plt.figure(figsize=figsize)
    p_vals = np.arange(-5,5,0.01)
    plt.plot(p_vals, L(p_vals), label=r'$L({\rm p})$')
    plt.plot(p_vals, grad_L(p_vals), label=r'$\nabla L({\rm p})$')
    plt.plot(p_vals, grad_L1(p_vals), label=r'$\nabla L_1({\rm p})$')
    plt.plot(p_vals, grad_L2(p_vals), label=r'$\nabla L_2({\rm p})$')
    
    plt.legend()
    plt.grid()
    plt.xlabel(r"${\rm p}$")
    plt.tight_layout()
    plt.savefig("./Figure11_5a.pdf")
    plt.show()

def figure11_5b( N_k = 2000):
    """
    Runs a SGD simulation to solve L(p)=0.

    Parameters:
        N_k : simulation time length.
    """
        
    figsize = config.global_config(type= 1)
    C = [1.0, 1.0, 1.0]         # constants for step size
    alpha = [0.4, 0.8, 1.2]     # decay rate for step size
    N_setting = len(C)          # Number of settings

    p_list = np.zeros( ( N_setting , N_k+1 ) )
    y_list = np.zeros( ( N_setting , N_k ) )
    plt.figure(figsize=figsize)  
    p_ini = 1                   # initial value for SGD
    p_list[:,0] = np.ones_like(p_list[:,0]) * p_ini 
    for setting in range( N_setting ):
        for k in range(N_k):        
            p = p_list[setting,k]
            y_list[setting, k] = np.random.choice([grad_L1, grad_L2], p=[0.5, 0.5])(p)
            p_list[setting,k+1] = p - C[setting]/((k+1)**alpha[setting]) * y_list[setting,k]
        plt.plot(p_list[setting,:],label=r"$\alpha={}$".format(alpha[setting]))

    plt.scatter(0,p_ini,marker='o',s=40,clip_on=False,color='black',label= "Initial Value",zorder=20)
    plt.ylabel(r"$p_k$")
    plt.xlabel(r"$k$")
    plt.xlim([0,N_k])
    plt.ylim([-2,2])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure11_5b.pdf")
    plt.show()

if __name__ == '__main__':
    figure11_5a()
    figure11_5b()

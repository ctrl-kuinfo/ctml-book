# Author: Kenji Kashima
# Date  : 2025/09/10
# Note  : You should install scipy first.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.ticker import MultipleLocator
import config

np.random.seed(13)


# P(i,j) = Prob(current state = j → next state = i)
P = np.array([[1/3,    1/3,      0,      0],
                [0  ,    1/3,    1/3,      0],
                [0  ,    1/3,    1/3,    1/3],
                [2/3,      0,    1/3,    2/3]])
m,n = P.shape

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    stationary distribution vector of stochastic matrix P
    """
    eigvals, eigvecs = np.linalg.eig(P)  # right eigenvector
    index = np.argmax(np.real(eigvals))    # index for eig 1

    eigenvector_for_1 = np.real(eigvecs[:, index])
    # normalization
    p_stationary = eigenvector_for_1 / np.sum(eigenvector_for_1)
    return p_stationary

def figure10_2b(k_bar= 300):
    figsize = config.global_config(type= 1)
    # Figure 10.2 (a) transition probability
    
    p_stationary = stationary_distribution(P)
    print("p_stationary for P^0：", p_stationary)

    # p_stable = np.ones((4,1))/4
    # for _ in range(100):
    #     p_stable =  P@p_stable
    # print("p_100=",p_stable)
    
    # accumulated transition probability
    # P_accum(i,j) = Prob(current state = j → next state <= i)
    P_accum = np.cumsum(P, axis=0)

    state_list = np.zeros(k_bar, dtype=np.int64) 
    state_list[0] = 4 # start at 4

    # simulation of autonomous Markov Chain
    for k in range(1,k_bar):
        u = np.random.rand()                    # uniform distribution on [0,1]
        T = P_accum[:,int(state_list[k-1])-1]   # T(i) = Prob( next state <= i )
        for i in range(m):
            if u <= T[i]:
                state_list[k] = i+1
                break

    plt.figure(figsize=figsize)
    plt.stairs(state_list,linewidth=2,zorder = 10,baseline=4)
    plt.ylim([0.8,4.2])
    plt.xlim([0,k_bar])
    plt.xticks(np.arange(0,k_bar+1,10))
    plt.xlabel(r"$k$")
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("./Figure10_2b.pdf")
    plt.show()


def L_KL(v, beta, cost, P):
    """ L_KL(v) % find the solution of equation (10.20)"""
    return np.sum((v - cost + np.log(P.T @ np.exp(- beta*v)))**2)
         
def L_IRL(v, beta, a, b, P):
    """ L_IRL(v)  % find the solution of equation below (10.29)"""
    return beta * a @ v + b @ np.log(P.T @ np.exp(-beta * v))

def figure10_4(k_bar= 1000, sigma = 1.0):
    
    beta = 0.8
    invbeta = 1/beta
    cost = np.array([1, 2, 3, 4]) * sigma

    # find optimal transition probability P^pi
    results = minimize( fun = L_KL,
                        x0 = np.ones(4)*5,
                        args = (beta, cost, P),
                        method = "L-BFGS-B",
                        options = {
                            'gtol': 1e-10,    
                            'ftol': 1e-14,    
                            'maxiter': 1000, 
                            'maxls': 50       
                        }
                    )
    print("error:",results.fun)
    print("V*:",results.x)
    z_opt = np.exp(-beta*results.x) 
    P_opt = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            P_opt[i,j] = P[i,j] / (P[:,j]@(z_opt/z_opt[i]))
    
    print("sigma:",sigma, "Ppi:\n", P_opt)

    p_opt_stationary = stationary_distribution(P_opt)
    print("p_stationary for P_opt：", p_opt_stationary)

    P_accum = np.cumsum(P_opt, axis=0)

    state_list = np.zeros(k_bar+1,dtype=np.int8) 
    state_list[0] = 4   # start at 4
    a, b=np.array([0,0,0,1]), np.array([0,0,0,0])   # state visitation counts
    inv_l = np.zeros(n) # estimated state cost
    inv_l_hist =np.zeros((4,k_bar))   # history of estimated state costs

    # Solve L_IRL = 0 with V[1] = 'offset'
    offset = 0

    for k in range(k_bar):
        results = minimize(fun=L_IRL,x0= np.array([3,4,10,10]), args=(beta,a,b,P),
                        constraints= {'type': 'eq', 'fun': lambda x: x[0]-offset} )
        results_v = results.x   # estimated V*
        inv_z_opt = np.exp( -beta*results_v )   # estimated Z* by eq. (10.21)  
        inv_l = results_v + np.log(P.T @ inv_z_opt)   # estimated l eq. (10.25)
        inv_l_hist[:,k] = inv_l # store history of estimated state costs

        # simulation of optimally controlled Markov Chain
        u = np.random.rand()                
        T = P_accum[:,int(state_list[k])-1]     
        for i in range(m):
            if u <= T[i]:
                state_list[k+1] = i+1
                break
        # state visitation counts
        a[state_list[k+1]-1] +=1 
        b[state_list[k]-1] +=1

    figsize = config.global_config(type=1)
    plt.figure(figsize=figsize)
    plt.stairs(state_list,linewidth=2,zorder = 10,baseline=4)
    plt.ylim([0.8,4.2])
    plt.xlim([0,50])
    plt.xticks(np.arange(0,50+1,10))
    
    plt.xlabel(r"$k$")
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("./Figure10_4a.pdf")

    figsize = config.global_config(type=1)
    plt.figure(figsize=figsize)
    
    # state cost for state 1 is fixed to 1
    inv_l_hist = inv_l_hist - inv_l_hist[0,:] + 1
    for i in range(4):
        plt.plot(inv_l_hist[i,:],label=r"$\ell_{}$".format(i+1))
    plt.ylim([0,6.5])
    plt.yticks(np.arange(0,6,1))
    plt.xlim([0,k_bar])
    plt.xlabel(r"$k$")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("./Figure10_4b.pdf")
    plt.show()

if __name__ == '__main__':
    figure10_2b(k_bar=50)
    figure10_4(k_bar=1000,sigma=1.0)

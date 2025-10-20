# Author: Kenji Kashima
# Date  : 2025/09/01

import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(24)

# System parameters setup
def initialization(k_bar=1000):
    # ARX parameters
    a = [1.2, -0.47, 0.06]  # coefficient of (z-0.5)(z-0.4)(z-0.3)
    b = [1.0, 2.0]          # b2, b1

    q0 = np.array([0,0,0,1,1]) # initial states and inputs, note that u0=u1=1

    p0 = np.array([0,0,0,0,0]) # initial parameters are zeros
    Sigma0 = 1e4 * np.eye(len(p0))      # initial covariance

    return a, b, q0, p0, Sigma0, k_bar


def sysid_module(p_star:np.ndarray, a_dim:int, q0:np.ndarray, u:np.ndarray, v:np.ndarray, p0:np.ndarray, Sigma0:np.ndarray, alpha:float):
    """
    % System identification
    % args: p_star -- the TRUE parameters sequence
    %       a_dim    -- the number of a  (b_dim denotes the number of b)
    %       q0     -- initial data [y_n,...,y_0,u_m,...,u_0]
    %       u      -- u_data
    %       v      -- v_data
    %       p0     -- initial parameters 
    %       Sigma0 -- initial Sigma in Algorithm 3
    %       alpha  -- forgetting factor
    """

    # k_dat is the data length (usually, =k_bar)
    k_dat, p_dim = np.shape(p_star) # p_dim = b_dim + a_dim 

    y = np.zeros(k_dat-1) 
    q = np.zeros((k_dat,p_dim))
    q[0,:] = q0

    # open loop simulation to genereate y data
    # the true system y_k = q_k^T * p^* + v_k
    # where q_k = [y_{k-1}, y_{k-2}, y_{k-3}, u_{k-1}, u_{k-2}]
    for k in range(k_dat-1):
        y[k] =  p_star[k,:] @ q[k,:] + v[k]
        q[k+1,:] = np.hstack([y[k],q[k,:a_dim-1],u[k],q[k,a_dim:p_dim-1]])

    p_est = np.zeros((k_dat,p_dim))
    a_err = np.zeros(k_dat)
    b_err = np.zeros(k_dat)
    TrSigma = np.zeros(k_dat)

    p_est[0,:] = p0 
    Sigma = Sigma0

    # squared norm of estimation error vector 
    a_err[0] = np.sum((p_star[0, :a_dim] - p0[:a_dim])**2)
    b_err[0] = np.sum((p_star[0, a_dim:] - p0[a_dim:])**2)
    TrSigma[0] = np.trace(Sigma)

    for k in range (k_dat-1):
        # Algorithm 3:
        H = Sigma@q[k,:] / (alpha + q[k,:]@Sigma@q[k,:])        # line 1
        p_est[k+1,:] = p_est[k,:]+H*(y[k]-p_est[k,:]@q[k,:])    # line 2
        Sigma = (Sigma - np.outer(H, q[k,:] ) @ Sigma) / alpha  # line 3
        Sigma = 0.5 * (Sigma + Sigma.T) # to avoid numerical instability
 
        # squared norm of estimation error vector 
        a_err[k+1] = np.sum((p_star[k+1, :a_dim] - p_est[k+1, :a_dim])**2)
        b_err[k+1] = np.sum((p_star[k+1, a_dim:] - p_est[k+1, a_dim:])**2)
        TrSigma[k+1] = np.trace(Sigma)      # trace of Sigma
    return a_err, b_err, TrSigma, p_est, y

def figure8_2a(k_bar=1000):
    """
        k_bar - number of k
    """
    alpha = 1
    a, b, q0, p0, Sigma0, k_bar = initialization(k_bar)
    p_star = np.array([a+b]).repeat(k_bar,axis=0) # p^* the true parameter
    a_dim = len(a)

    figsize = config.global_config(type= 0)
    plt.figure(figsize=figsize)

    # u_k = 1, v_k~N(0,1)
    sigma_v = 1
    v = np.random.randn(k_bar)*sigma_v  # random noise N(0,sigma_v^2)
    u = np.ones(k_bar)                  # constant input u_k = 1 
    _,_,_,_,y = sysid_module(p_star,a_dim,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(y,label=r'$u_k=1,v_k\sim {\mathcal N}(0,1)$',linewidth=0.5)

    # u_k ~ N(1,1), v_k~N(0,1)
    sigma_v = 0.1
    v = np.random.randn(k_bar)*sigma_v # random noise N(0,sigma_v^2)
    u = np.ones(k_bar) + np.random.randn(k_bar)  # random input N(1,1)
    _,_,_,_,y = sysid_module(p_star,a_dim,q0,u,v,p0,Sigma0/sigma_v,alpha)

    plt.plot(y,label=r'$u_k \sim {\mathcal N}(1,1), v_k\sim {\mathcal N}(0,0.01)$',linewidth=0.5)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$y_k$")
    plt.xlim([0,k_bar])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure8_2a.pdf")
    plt.show()

def figure8_2b(k_bar=5000):
    """
        k_bar - number of k
    """
    a, b, q0, p0, Sigma0, k_bar = initialization(k_bar)
    a_dim = len(a)

    # p^* changes between a and a_changed
    a_changed = [1.0,-0.47,0.06] 
    p_star = np.empty((k_bar, len(p0)))
    mask = (np.arange(k_bar) % 2000 > 1000)
    p_star[mask] = a_changed + b
    p_star[~mask] = a + b
    
    # u_k ~ N(0,1), v_k ~ N(0,1)
    sigma_v = 0.1
    v = np.random.randn(k_bar)*sigma_v # random noise N(0,sigma_v^2)
    u = np.random.randn(k_bar)         # constant input u_k ~ N(0,1) 

    figsize = config.global_config(type= 0)
    plt.figure(figsize=figsize)

    alpha = 1.0
    _,_,_,p_est,_ = sysid_module(p_star,a_dim,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(p_est[:,0],label=r'$\alpha=1$',linewidth=0.5)

    alpha = 0.995
    _,_,_,p_est,_ = sysid_module(p_star,a_dim,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(p_est[:,0],label=r'$\alpha=0.995$',linewidth=0.5)
  
    alpha = 0.8
    _,_,_,p_est,_ = sysid_module(p_star,a_dim,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(p_est[:,0],label=r'$\alpha=0.8$',linewidth=0.5)
    
    plt.plot(p_star[:,0],label = "True")

    plt.xlabel(r"$k$")
    plt.ylabel(r"${\rm a}_1$")
    plt.xlim([0,k_bar])
    plt.ylim([0.8,1.4])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure8_2b.pdf")
    plt.show()

def figure8_2cdef(k_bar=100000, label = "c"):
    """
        k_bar - number of k
    """
    a, b, q0, p0, Sigma0, k_bar = initialization(k_bar)
    a_dim = len(a)

    alpha = 1.0
    p_star= np.array([a+b]).repeat(k_bar,axis=0) # p^* the true parameter
    
    if label == "c" or label == "d":
        sigma_v = 1
    else:
        sigma_v = 0.1

    if label == "c" or label == "e":
        # u_k = 1, v_k~N(0,sigma_v^2)
        v = np.random.randn(k_bar)*sigma_v # random noise N(0,sigma_v^2)
        u = np.ones(k_bar)         # constant input u_k ~ N(0,1) 
    else:
        # u_k ~ N(1,1), v_k~N(0,sigma_v^2)
        v = np.random.randn(k_bar)*sigma_v # random noise N(0,sigma_v^2)
        u = np.ones(k_bar) + np.random.randn(k_bar)         # constant input u_k ~ N(0,1) 

    figsize = config.global_config(type= 0)
    plt.figure(figsize=figsize)
    a_err,b_err,TrSigma,_,_ = sysid_module(p_star,a_dim,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.loglog(a_err,label=r'$\|\check{\rm p}^a - {\rm a}^*\|^2$',linewidth=1)
    plt.loglog(b_err,label=r'$\|\check{\rm p}^b - {\rm b}^*\|^2$',linewidth=1)
    plt.loglog(TrSigma*sigma_v,label=r'${\rm Trace}(\check\Sigma)$',linewidth=1)

    plt.xlabel(r"$k$")
    plt.xlim([1,k_bar])
    plt.ylim([1e-10,1e5])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure8_2{}.pdf".format(label))

    plt.show()


if __name__ == '__main__':
    figure8_2a(k_bar=1000)
    figure8_2b(k_bar=5000)
    figure8_2cdef(k_bar=100000,label="c")
    figure8_2cdef(k_bar=100000,label="d")
    figure8_2cdef(k_bar=100000,label="e")
    figure8_2cdef(k_bar=100000,label="f")

# Author: Kenji Kashima
# Date  : 2025/09/01

import numpy as np
import matplotlib.pyplot as plt
import config

seed = 13

def gaussian_kernel(x,y,c):
    """
    Gaussian kernel function (RBF kernel) in eq 12.7 and 12.8. 

    Parameters:
        x : ndarray
        y : ndarray
        c : float, Kernel width (scale parameter).

    Returns : ndarray
            Kernel matrix K where K[i,j] = exp( - (x_i - y_j)^2 / c^2 )
    """
    x = np.asarray(x).reshape(-1, 1)   # (N,1)
    y = np.asarray(y).reshape(-1, 1)   # (M,1)
    return np.exp(-((x - y.T)**2) / (c**2))

def min_kernel(x,y):
    """
    Min kernel function in eq. 12.30 and 12.8.

    Parameters:
        x : ndarray
        y : ndarray
    
    Returns : ndarray
            Kernel matrix K where K[i,j] = min(x_i, y_j).
    """
    x = np.asarray(x).reshape(-1, 1)  
    y = np.asarray(y).reshape(-1, 1)  
    return np.minimum(x, y.T)         

def figure12_2(N_x= 100, c = 0.1, label="a"):
    '''  
        Figure12.2(a) c = 0.1 
        Figure12.2(b) c = 1.0
    '''
    figsize = config.global_config(type= 1)
    x = np.linspace(0, 1, N_x)                              # x for plot (query)          
    K = gaussian_kernel(x,x,c)     

    # prior distribution
    prior_mean = x # y=mu(x)=x

    plt.figure(figsize=figsize)

    for i in range(5): # 5 samples
         plt.plot(x,np.random.multivariate_normal(prior_mean,K),color="gray",alpha=0.8)  #eq 12.13
    plt.plot(x,prior_mean,label=r"$\mu({\rm x})$")  
    x_SD = np.sqrt(np.diag(K))   # standard deviation of x

    plt.fill_between(x,prior_mean-x_SD,prior_mean+x_SD,color="blue",alpha=0.25)
    plt.xlim([0,1])
    plt.ylim([-3,3])
    plt.xlabel(r"${\rm x}$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure12_2{}.pdf".format(label))
    plt.show()

def figure12_3and4(N_x= 100, s_bar = 10, c = 0.1, label="a", kernel="Gaussian"):
    '''  
        Figure12.x(a) s_bar=10 
        Figure12.x(b) s_bar=50
    '''
    np.random.seed(seed)
    figsize = config.global_config(type= 1)
    x_data = np.random.rand(s_bar) * np.random.rand(s_bar)  # x data points
    x = np.linspace(0, 1, N_x)                              # x for plot (query)          
        
    # kernel function selection
    if kernel == "Gaussian":
        K = gaussian_kernel(x_data, x_data, c)  # eq 12.8
        Ktmp = gaussian_kernel(x, x, c)
        kx = gaussian_kernel(x,x_data, c)
        title = '3'+label    # eq 12.9
    else:
        K = min_kernel(x_data, x_data)  # eq 12.8
        Ktmp = min_kernel(x, x)
        kx = min_kernel(x,x_data)    # eq 12.9
        title = '4'+label
    # K = (K + K.T)/2  # to avoid numerical asymmetry

    # function in example 12.1.9 and Fig 12.4
    mu_x = lambda x : x                     # mu(x)=x
    fn_x = lambda x :np.sin(4*np.pi*x)    # y(x) = sin(4πx)

    fn_list = fn_x(x)

    sigma = 0.1
    e_s = sigma * np.random.randn(s_bar)    # noise sample
    y_data = fn_x(x_data) + e_s         # observation data 

    plt.figure(figsize=figsize)
    plt.plot(x,mu_x(x),color = 'red' ,label=r"$\mu({\rm x})$") 
    plt.plot(x,fn_list,'k--',label=r'$\sin(4\pi {\rm x})$');  

    #plot samples
    plt.scatter(x_data,y_data,edgecolor='b',marker='o',facecolor='none', label=r"${\rm y}_s$")

    prior_mean = mu_x(x_data)
    mean = mu_x(x)+kx @ np.linalg.solve(K + sigma**2 * np.eye(s_bar) , y_data-prior_mean ) # eq 12.16
    Kpost = Ktmp - kx @ np.linalg.solve(K + sigma**2 * np.eye(s_bar) , kx.T )  # eq 12.17
    x_SD = np.sqrt(np.diag(Kpost))  # posterior standard deviation

    plt.fill_between(x,mean-x_SD,mean+x_SD,color="blue",alpha=0.25)
    plt.plot(x,mean,"-.", color="blue",   label = r"$\mu ({\rm x}|\mathcal D)$")
    plt.xlim([0,1])
    plt.ylim([-1.5,3.0])
    plt.yticks([-1,0,1,2,3])

    plt.xlabel(r"${\rm x}$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure12_{}.pdf".format(title))
    plt.show()


if __name__ == '__main__':
    figure12_2(N_x=100, c = 0.1, label="a")                             #Figure12.2(a)
    figure12_2(N_x=100, c = 1.0, label="b")                             #Figure12.2(b)
    figure12_3and4(N_x=100, s_bar = 10, label="a", kernel="Gaussian")   #Figure12.3(a)
    figure12_3and4(N_x=100, s_bar = 50, label="b", kernel="Gaussian")   #Figure12.3(b)
    figure12_3and4(N_x=100, s_bar = 10, label="a", kernel="min")        #Figure12.4(a)
    figure12_3and4(N_x=100, s_bar = 50, label="b", kernel="min")        #Figure12.4(b)

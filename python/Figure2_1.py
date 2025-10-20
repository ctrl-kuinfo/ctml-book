# Author: Kenji Kashima
# Date  : 2025/06/21

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
import config

def figure2_1a():
    figsize = config.global_config(type=1)
    x = np.linspace(0, 10, 1000)
    
    # PDF for N(0, 2)
    pdf_x2 = norm.pdf(x, 0, np.sqrt(2))
    
    # PDF for Laplace(0, 1) 
    pdf_x3 = laplace.pdf(x, 0, 1)

    plt.figure(figsize=figsize)
    plt.plot(np.abs(x), pdf_x2, label=r'$\mathcal{N}(0, 2)$', color='orange')
    plt.plot(np.abs(x), pdf_x3, label=r'${\rm Lap}(0, 1)$', color='purple')

    plt.legend()
    plt.xlabel(r'$\vert {\rm x}\vert$')
    plt.xlim([0,10])
    plt.ylim([1e-12,1])
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Figure2_1a.pdf")
    plt.show()

def figure2_1b():
    figsize = config.global_config(type=1)
    x = np.linspace(-10, 10, 1000)

    # PDFs for N(-2, 1) and N(2, 1)
    pdf_x1 = norm.pdf(x, -2, 1)
    pdf_x2 = norm.pdf(x, 2, 1)

    # PDF for the sum (x1 + x2), which is N(0, 2)
    pdf_x1_plus_x2 = norm.pdf(x, 0, np.sqrt(2))

    # PDF for the mixture 0.5*N(-2, 1) + 0.5*N(2, 1) 
    pdf_mixture = 0.5 * norm.pdf(x, -2, 1) + 0.5 * norm.pdf(x, 2, 1)

    # PDF for Laplace(0, 1) 
    pdf_x3 = laplace.pdf(x, 0, 1)

    plt.figure(figsize=figsize)
    plt.plot(x, pdf_x1, label=r'$x_1 \sim \mathcal{N}(-2, 1)$', color='blue')
    plt.plot(x, pdf_x2, label=r'$x_2 \sim \mathcal{N}(2, 1)$', color='orange')
    plt.plot(x, pdf_x1_plus_x2, label=r'$(x_1+x_2) \sim \mathcal{N}(0, 2)$', color='green')
    plt.plot(x, pdf_mixture, label=r'$(\varphi_{x_1}+\varphi_{x_2})/2$', color='red')
    plt.plot(x, pdf_x3, label=r'$x_3\sim {\rm Lap}(0, 1)$', color='purple')

    plt.legend()
    plt.xlabel(r'${\rm x}$')
    plt.xlim([-5,5])
    plt.ylim([0,1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Figure2_1b.pdf")
    plt.show()

if __name__ == '__main__':
    figure2_1a()
    figure2_1b()
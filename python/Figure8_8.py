# Author: Kenji Kashima
# Date  : 2025/09/11

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
import config

np.random.seed(1)
k_bar = 200
n_sample = 5
x_dim = 5

A = np.random.rand(x_dim, x_dim)
eigvals = np.linalg.eigvals(A)
A = A / np.max(np.abs(eigvals)) * 0.95
B = np.random.rand(x_dim, 1)
X = dlyap(A, B @ B.T)

figsize = config.global_config(type=1)
fig = plt.figure(figsize=(15, 10))   
ax = fig.add_subplot(111, projection="3d")
ax.grid(True)

x_list = {}
for s in range(1, n_sample + 1):
    x_list[s] = np.random.multivariate_normal(np.zeros(x_dim), X).reshape(-1, 1)
    for t in range(k_bar):
        new_col = A @ x_list[s][:, t].reshape(-1, 1) + B * np.random.randn()
        x_list[s] = np.hstack([x_list[s], new_col])
    k_vals = np.arange(-k_bar // 2, k_bar // 2 + 1)
    ax.plot3D(k_vals, x_list[s][0, :], x_list[s][1, :], linewidth=1)

ax.set_xlim([-k_bar // 2, k_bar // 2])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
plt.grid()

ax.set_xlabel(r"$k$", labelpad=50)
ax.set_ylabel(r"$({\rm y})_1$", labelpad=10)
ax.set_zlabel(r"$({\rm y})_2$", labelpad=10)

ax.view_init(elev=10, azim=75)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_box_aspect([3, 2, 1])   

plt.savefig("./Figure8_8.pdf", bbox_inches="tight", pad_inches=1)
plt.show()
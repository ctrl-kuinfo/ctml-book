# Author: Kenji Kashima
# Date  : 2025/09/01

import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(23)

directions = [(1,0), (-1,0), (0,1), (0,-1)]  # Right, Left, Up, Down

def init_value():
    g = grid_size  # 100
    I, J = np.ogrid[0:g+1, 0:g+1]  # 形状: (g+1,1), (1,g+1) → ブロードキャスト
    value = (
        1.5*np.exp(-((I-20)**2)/1000 - ((J-20)**2)/1000) +
        np.exp(   -((I-20)**2)/1000 - ((J-90)**2)/500 ) +
        np.exp(   -((I-40)**2)/1000 - ((J-50)**2)/500 ) +
        np.exp(   -((I-70)**2)/300  - ((J-70)**2)/500 ) +
        1.5*np.exp(-((I-80)**2)/300  - ((J-40)**2)/500 ) +
        np.exp(   -((I-50)**2)/800  - ((J-50)**2)/800 ) +
        1.2*np.exp(-((I-80)**2)/200  - ((J-20)**2)/200 ) +
        np.exp(   -((I-90)**2)/200  - ((J-10)**2)/200 )
    ) / 10.0
    return value

def all_agents_locally_optimal(state, value, sensor_range):
    IDn = state.shape[0]
    grid_size = value.shape[0]

    for ID in range(IDn):
        x0, y0 = state[ID, 0], state[ID, 1]
        base_value = sum_value_near_agents(state, sensor_range, x0, y0, value)

        for dx, dy in directions:
            proposed = state.copy()
            proposed[ID, 0] = np.clip(x0 + dx, 0, grid_size)
            proposed[ID, 1] = np.clip(y0 + dy, 0, grid_size)

            new_value = sum_value_near_agents(proposed, sensor_range, x0, y0, value)

            if new_value > base_value:
                return False  # 少なくとも1方向に改善できる → 局所最適ではない

    return True  # 全方向に対して改善できない → 局所最適

# Define the simulation function
def simulation(value, ini_state, Tmax, deterministic):
    """
    Runs a simulation of IDn agents moving in a 2D grid based on local value estimations.

    Parameters:
        value (ndarray): (101, 101) value map.
        Tmax (int): maximum simulation steps.
        deterministic (bool): if True, always move if estimated value improves.

    Returns:
        ini_state (ndarray): initial agent positions.
        state (ndarray): current agent position (shape: [IDn, 2]).
        x_list (ndarray): trajectory of x-coordinates (shape: [IDn, k]).
        y_list (ndarray): trajectory of y-coordinates (shape: [IDn, k]).
    """

    x_list = np.zeros((IDn, Tmax))
    y_list = np.zeros((IDn, Tmax))

    state = ini_state.copy()

    for k in range(Tmax):
        x_list[:, k] = state[:, 0]
        y_list[:, k] = state[:, 1]

        # Randomly select an agent and movement direction
        agent_id = np.random.randint(IDn)
        direction = directions[np.random.randint(4)]
        candidate_state = state.copy()

        # Apply movement in chosen direction
        candidate_state[agent_id, 0] = np.clip(candidate_state[agent_id, 0] + direction[0], 0, grid_size)
        candidate_state[agent_id, 1] = np.clip(candidate_state[agent_id, 1] + direction[1], 0, grid_size)

        # Evaluate local values
        x, y = state[agent_id, 0], state[agent_id, 1]
        current_value = sum_value_near_agents(state, sensor_range, x, y, value)
        candidate_value = sum_value_near_agents(candidate_state, sensor_range, x, y, value)

        if deterministic:
            if candidate_value > current_value:
                state = candidate_state
                print(f"Step {k}: Agent {agent_id} moved (deterministic)")

                if all_agents_locally_optimal(state, value, sensor_range):
                    x_list = x_list[:, :k+1]
                    y_list = y_list[:, :k+1]
                    print(f"All agents are locally optimal at step {k}. Simulation terminated.")
                    break
        else:
            prob_move = 1 / (1 + np.exp(beta * (current_value - candidate_value)))
            if np.random.rand() < prob_move:
                state = candidate_state
                print(f"Step {k}: Agent {agent_id} moved (stochastic)")

    return state, x_list[:, :k+1], y_list[:, :k+1]

def sum_value_near_agents(state, sensor_range, X, Y, value):
    r2 = sensor_range*sensor_range
    g = grid_size

    # evaluation range
    evaluation_range = sensor_range + 3
    X = int(X); Y = int(Y)
    i0 = max(0, X - evaluation_range)
    i1 = min(g, X + evaluation_range)
    j0 = max(0, Y - evaluation_range)
    j1 = min(g, Y + evaluation_range)

    # Sub-grid coordinates (shape: (H,1), (1,W))
    Ii, Jj = np.ogrid[i0:i1+1, j0:j1+1]  # H×1, 1×W

    # state: (IDn,2) represents (x,y) of each agent in the grid
    ax = state[:, 0][:, None, None]
    ay = state[:, 1][:, None, None]
    dist2 = (ax - Ii)**2 + (ay - Jj)**2           # shape: (IDn, H, W)

    # If any agent is within range (min distance) → boolean mask (H,W)
    near_mask = (dist2.min(axis=0) < r2)

    # Sum values of cells where mask is active
    return float(value[i0:i1+1, j0:j1+1][near_mask].sum())



def Figure10_5a(value):
    figsize = config.global_config(type= 1)
    # Figure 10.5(a): Contour plot of the value function
    plt.figure(figsize=figsize)

    plt.contourf(value.T, 20)
    plt.colorbar()
    plt.xlim([0, grid_size])
    plt.ylim([0, grid_size])
    plt.tight_layout()
    plt.savefig("./Figure10_5a.pdf")
    plt.show()

def Figure10_5b(value, Tmax=20000):
    figsize = config.global_config(type= 1)
    # Figure 10.5(b): Run simulation with deterministic=False
    state, x_list, y_list = simulation(value, ini_state, Tmax, False)
    plt.figure(figsize=figsize)
    plt.plot(ini_state[:, 0], ini_state[:, 1], '*', markersize=10)
    plt.plot(state[:, 0], state[:, 1], 'o', markersize=10)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(IDn):
        plt.plot(x_list[i, :], y_list[i, :])
        plt.plot(10 * np.sin(t) + state[i, 0], 10 * np.cos(t) + state[i, 1])
    plt.grid(True)
    plt.xlim([0,grid_size])
    plt.ylim([0,grid_size])
    plt.tight_layout()
    plt.savefig("./Figure10_5b.pdf")
    plt.show()

def Figure10_5c(value, Tmax=20000):
    figsize = config.global_config(type= 1)
    # Figure 10.5(c): Run simulation with deterministic=True
    state, x_list, y_list = simulation(value, ini_state, Tmax, True)
    plt.figure(figsize=figsize)
    plt.plot(ini_state[:, 0], ini_state[:, 1], '*', markersize=10)
    plt.plot(state[:, 0], state[:, 1], 'o', markersize=10)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(IDn):
        plt.plot(x_list[i, :], y_list[i, :])
        plt.plot(10 * np.sin(t) + state[i, 0], 10 * np.cos(t) + state[i, 1])
    plt.grid(True)
    plt.xlim([0,grid_size])
    plt.ylim([0,grid_size])
    plt.tight_layout()
    plt.savefig("./Figure10_5c.pdf")
    plt.show()

if __name__ == '__main__':
    grid_size = 100
    IDn = 12
    sensor_range = 8
    beta = 1
    ini_state = np.ones((IDn, 2)) * 10
    
    value = init_value( )
    Figure10_5a(value)
    Figure10_5b(value)
    Figure10_5c(value)

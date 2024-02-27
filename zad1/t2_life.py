#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/nfs/work/2022/tp449528/lib")
from numba import jit

@jit(nopython = True)
def evlove(n_val, curr):
    if n_val == 3:
        return 1
    elif n_val == 2 and curr == 1:
        return 1
    else:
        return 0
print("xd")
#TASK 2
#init
life_shape = (256,512)
L = life_shape[1]
life = np.random.randint(0,2,life_shape)
T_max = 1000

print("xdxd")
#%%
def save(time, life, life_shape):
    fig = life.reshape(life_shape)
    plt.imshow(fig, interpolation = 'none')
    plt.savefig(f"./zad1/imgs/life_{time}.png")

@jit(nopython = True)
def calc_neighbour(life,L):
    neighbour_val = np.roll(life,1) + np.roll(life,-1) \
            + np.roll(life,L) + np.roll(life,-L) \
            + np.roll(life,L+1) + np.roll(life,L-1)\
            + np.roll(life,-L+1) + np.roll(life,-L-1)
    return neighbour_val


def run(time, life,L, life_shape):
    for t in range(time):
        n_vals = calc_neighbour(life,L)

        life = np.array(list(map(evlove, n_vals, life)))
        if t%100 == 0:
            save(t,life, life_shape)

run(T_max, life.ravel(), L, life_shape)
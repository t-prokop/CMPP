#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/nfs/work/2022/tp449528/lib")
from numba import jit
from itertools import repeat

rule_str = input("rule:")

sustain_rule, born_rule = rule_str.split('/')
sustain_rule = list(map(int,sustain_rule))
born_rule = list(map(int,born_rule))

@jit(nopython = True)
def evlove(pos, L, n_val, curr, born_rule, sustain_rule):
    if pos//L == 0 or (pos%L)==0 or n_val in born_rule or (n_val in sustain_rule and curr == 1):
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


def run(time, life,L, life_shape, tot_length, born_rule, sustain_rule):
    for t in range(time):
        n_vals = calc_neighbour(life,L)

        life = np.array(list(map(evlove, np.arange(tot_length),repeat(tot_length), n_vals, life, repeat(born_rule), repeat(sustain_rule))))
        if t%100 == 0:
            save(t,life, life_shape)

run(T_max, life.ravel(), L, life_shape, life_shape[0]*life_shape[1], born_rule, sustain_rule)
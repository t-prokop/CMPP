#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/nfs/work/2022/tp449528/lib")
import numba

def evlove(n_val, curr):
    if n_val == 3:
        return 1
    elif n_val == 2 and curr == 1:
        return 1
    else:
        return 0

#TASK 2
#init
life_shape = (256,512)
life = np.random.randint(0,2,life_shape)
T_max = 1000
#%%
for t in range(T_max):
    # plt.imshow(life)
    right_roll = np.roll(life,1,1)
    left_roll = np.roll(life,-1,1)

    neighbour_val = np.roll(life,1,0) + np.roll(life,-1,0) \
        + right_roll + left_roll \
        + np.roll(right_roll,1,0) + np.roll(right_roll,-1,0)\
        + np.roll(left_roll,1,0) + np.roll(left_roll,-1,0)

    life = np.array(list(map(evlove, neighbour_val.ravel(), life.ravel()))).reshape(life_shape)
    plt.imshow(life, interpolation = 'none')
    plt.savefig(f"life_{t}.png")

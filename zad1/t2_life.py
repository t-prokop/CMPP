# %%
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/nfs/work/2022/tp449528/lib")

# TASK 2
# init
life_shape = (256, 512)
# life_shape = (8, 8)
L = life_shape[1]
life = np.random.randint(0, 2, life_shape)
T_max = 10000
boundary = np.zeros(life_shape)
for i in range(life_shape[0]):
    boundary[i, 0] = 1
    boundary[i, -1] = 1
for i in range(life_shape[1]):
    boundary[0, i] = 1
    boundary[-1, i] = 1
# %%


def save(time, life, filenames):
    plt.imshow(life, interpolation='none')
    plt.savefig(f"./imgs/life_{time}.png")
    filenames.append(f"./imgs/life_{time}.png")


@jit(nopython=True)
def calc_neighbour(life, L):
    life = life.ravel()
    neighbour_val = np.roll(life, -1) + np.roll(life, 1) + np.roll(life, L) + np.roll(
        life, -L) + np.roll(life, L+1) + np.roll(life, -L+1) + np.roll(life, L-1) + np.roll(life, -L-1)
    # neighbour_val = np.roll(life, (1,0), axis=(0,1)) + np.roll(life, (-1,0), axis=(0,1)) \
    #     + np.roll(life, (0,-1), axis=(0,1)) + np.roll(life, (0,1), axis=(0,1)) \
    #     + np.roll(life, (1,1), axis=(0,1)) + np.roll(life, (-1,1), axis=(0,1))\
    #     + np.roll(life, (-1,-1), axis=(0,1)) + np.roll(life, (1,-1), axis=(0,1))
    return neighbour_val


# @jit(nopython=True)
def calc_next_frame(life, L, boundary):
    n_vals = calc_neighbour(life, L)
    n_vals = n_vals.reshape(life.shape)
    newlife = (n_vals == 3) | ((n_vals == 2) & (life == 1)) | (boundary == 1)
    return newlife.astype(int)


def run(time, life, L, boundary, filenames):

    for t in range(time):
        if t % 100 == 0:
            save(t, life, filenames)

        life = calc_next_frame(life, L, boundary)

    save(t, life, filenames)


filenames = []
run(T_max, life, L, boundary, filenames)
# %%
import imageio
with imageio.get_writer('test.gif', mode='I') as writer:
    for frame in filenames:
        image = imageio.imread(frame)
        writer.append_data(image)
writer.close()

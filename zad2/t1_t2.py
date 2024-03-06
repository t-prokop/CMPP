# %%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
# world == 0 -> always cooperate
# world == 1 -> always defect
# cc (0) = 1
# cd (1) = 0
# dc (10)= b
# dd (11)= 0
# %%


def evolve_one_step(world, b):
    vals = calculate_play_values(world, b)
    evolved_world = choose_new_strat(world, vals)
    return evolved_world


def choose_new_strat(world, vals_world):
    rolls = [(1, 0), (0, 1), (-1, 0), (0, -1),
             (-1, 1), (1, -1), (-1, -1), (1, 1)]
    new_strats = np.copy(world)
    max_vals = np.copy(vals_world)

    for roll_dir in rolls:
        world_rolled = np.roll(world, roll_dir, axis=(0, 1))
        vals_world_rolled = np.roll(vals_world, roll_dir, axis=(0, 1))

        new_strats = np.where(vals_world_rolled > max_vals, world_rolled, new_strats)
        max_vals = np.where(vals_world_rolled > max_vals, vals_world_rolled, max_vals)

    return new_strats


def calculate_play_values(world, b):
    rolls = [(1, 0), (0, 1), (0, 0), (-1, 0), (0, -1),
             (-1, 1), (1, -1), (-1, -1), (1, 1)]
    new_world_vals = np.zeros((world.shape[0], world.shape[1]))

    for roll_dir in rolls:
        not_world_boolean = (world+1) % 2
        world_rolled = np.roll(world, roll_dir, axis=(0, 1))
        not_world_rolled_boolean = np.roll(not_world_boolean, 1, axis=0)
        new_world_vals += play(world, world_rolled,
                               not_world_boolean, not_world_rolled_boolean, b)

    return new_world_vals


def play(world, world_rolled, not_world, not_world_rolled, b):
    played = 10*world + world_rolled
    played_vals = b*(played == 10).astype(int) + (played == 0).astype(int)
    return played_vals


def show_world(min_val, max_val, num_colors, world, t, filenames, save=True):
    cmap = plt.get_cmap('RdBu', num_colors)
    mat = plt.matshow(world, vmin=min_val, vmax=max_val,
                      interpolation='none', cmap=cmap)
    cax = plt.colorbar(mat, ticks=np.arange(min_val, max_val + 1))
    if save:
        plt.savefig(f"./imgs/{t}.png")
        filenames.append(f"./imgs/{t}.png")
        plt.clf()
    else:
        plt.show()

# %%
# value assignments
L = 201
coop_val = 0
defect_val = 1
b = 1.9
T_max = 50
# %%
# generating worlds
# world = np.random.randint(coop_val, defect_val+1, (L, L))
world = np.zeros((L, L))
world[L//2, L//2] = 1

# condition_cooperator = [world!=coop_val, world == coop_val]
# choice_cooperator = [0, coop_val]

# condition_defector = [world!=defect_val, world == defect_val]
# choice_defector = [0, defect_val]

# cooperators_world = np.select(condition_cooperator, choice_cooperator)
# defectors_world = np.select(condition_defector, choice_defector)

# %%
filenames = []
for t in range(T_max):
    show_world(0, 1, 2, world, t, filenames)
    world = evolve_one_step(world, b)
show_world(0, 1, 2, world, t+1, filenames)
# %%
import imageio
with imageio.get_writer('t1.gif', mode='I') as writer:
    for frame in filenames:
        image = imageio.imread(frame)
        writer.append_data(image)
writer.close()

#%%
#T2
t2_world = np.concatenate((np.zeros(L**2//2),np.full(L**2 - L**2//2, 1)))
np.random.shuffle(t2_world)
t2_world = t2_world.reshape(L, L)
plt.imshow(t2_world, cmap='RdBu', interpolation='none')
plt.show()
#%%
b_vals = np.linspace(1.3, 2.7, 40)
freq_def_by_b = []
T_max = 50

for b in b_vals:
    for t in range(T_max):
        t2_world = evolve_one_step(t2_world, b)
    num_def = np.count_nonzero(t2_world)
    freq_def = num_def/(L**2)
    freq_def_by_b.append(freq_def)
#%%
plt.scatter(b_vals, freq_def_by_b)
#%%
b_vals_fine = np.linspace(1.8, 2.2, 20)
freq_def_by_b_fine = []
for b in b_vals_fine:
    for t in range(T_max):
        t2_world = evolve_one_step(t2_world, b)
    num_def = np.count_nonzero(t2_world)
    freq_def = num_def/(L**2)
    freq_def_by_b_fine.append(freq_def)
#%%
all_freq_def_by_b = np.concatenate(freq_def_by_b, freq_def_by_b_fine)
all_b_vals = np.concatenate(b_vals, b_vals_fine)
# %%
plt.scatter(all_b_vals, all_freq_def_by_b)
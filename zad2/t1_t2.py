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
        new_world_vals += play(world, world_rolled, b)

    return new_world_vals


def play(world, world_rolled, b):
    played = 10*world + world_rolled
    played_vals = b*(played == 10).astype(int) + (played == 0).astype(int)
    return played_vals


def show_world(min_val, max_val, num_colors, world, t, filenames, b, options, save=True, show = False):
    cmap = plt.get_cmap('RdBu', num_colors)
    mat = plt.matshow(world, vmin=min_val, vmax=max_val,
                      interpolation='none', cmap=cmap)
    cax = plt.colorbar(mat, ticks = np.arange(min_val, max_val+1))
    cax.set_ticks(list(options.values()), labels=list(options.keys()))
    
    if save:
        plt.title(f"b={b}")
        plt.savefig(f"./imgs/{t}.png")
        filenames.append(f"./imgs/{t}.png")
        plt.clf()
        plt.close()
    elif show:
        plt.show()

# %%
# value assignments
L = 201
coop_val = 0
defect_val = 3
cd_val = 2
dc_val = 1
all_options = {"coop":coop_val,
               "coop->defect": cd_val,
               "defect->coop": dc_val,
               "defect": defect_val}
init_options = {"coop":coop_val,
                "defect": defect_val
}
b = 2.08
T_max = 350
# %%
# generating worlds
# world = np.random.randint(coop_val, defect_val+1, (L, L))
world = np.zeros((L, L))
world[L//2, L//2] = 1
show_world(0,3,2,3*world,0,[],b,init_options,False,True)

# condition_cooperator = [world!=coop_val, world == coop_val]
# choice_cooperator = [0, coop_val]

# condition_defector = [world!=defect_val, world == defect_val]
# choice_defector = [0, defect_val]

# cooperators_world = np.select(condition_cooperator, choice_cooperator)
# defectors_world = np.select(condition_defector, choice_defector)

# %%
filenames = []
world_prev = np.zeros(world.shape)
for t in range(T_max):
    show_world(0, 3, 4, 2*world+world_prev, t, filenames, b, all_options)
    world_prev = np.copy(world)
    world = evolve_one_step(world, b)
show_world(0,3,2,3*world,0,[],b,init_options,False,True)
# %%
import imageio
with imageio.get_writer(f'single{b}.gif', mode='I') as writer:
    for frame in filenames:
        image = imageio.imread(frame)
        writer.append_data(image)
writer.close()


#%%
L=201
b_vals = [1.8743589743589744]
freq_def_by_b = []
T_max = 100

for b in b_vals:
    if b == 1.8743589743589744:
        print("xd")
        save = True
        filenames = []
    else:
        save = False
    t2_world = np.concatenate((np.zeros(L**2//2),np.full(L**2 - L**2//2, 1)))
    np.random.shuffle(t2_world)
    t2_world = t2_world.reshape(L, L)
    for t in range(T_max):
        if save:
            show_world(0,3,2,3*t2_world,t,filenames,b,init_options,save, False)
        t2_world = evolve_one_step(t2_world, b)
    if save:
        print(filenames)
        import imageio
        with imageio.get_writer(f'b_{b:.2f}.gif', mode='I') as writer:
            for frame in filenames:
                image = imageio.imread(frame)
                writer.append_data(image)
            writer.close()
    num_def = np.count_nonzero(t2_world)
    freq_def = 100*num_def/(L**2)
    freq_def_by_b.append(freq_def)
#%%
plt.scatter(b_vals, freq_def_by_b)
plt.xlabel("b value")
plt.ylabel("% of defectors")
plt.title(f"time of evolution: {T_max}")
#%%
b_vals_fine = np.linspace(1.7, 2.3, 50)
freq_def_by_b_fine = []
for b in b_vals_fine:
    t2_world = np.concatenate((np.zeros(L**2//2),np.full(L**2 - L**2//2, 1)))
    np.random.shuffle(t2_world)
    t2_world = t2_world.reshape(L, L)
    for t in range(T_max):
        t2_world = evolve_one_step(t2_world, b)
    num_def = np.count_nonzero(t2_world)
    freq_def = 100*num_def/(L**2)
    freq_def_by_b_fine.append(freq_def)
#%%
plt.scatter(b_vals_fine, freq_def_by_b_fine)
plt.xlabel("b value")
plt.ylabel("% of defectors")
plt.title(f"time of evolution: {T_max}")    
#%%
all_freq_def_by_b = freq_def_by_b + freq_def_by_b_fine
all_b_vals = np.concatenate((b_vals, b_vals_fine))
# %%
plt.scatter(all_b_vals, all_freq_def_by_b)
plt.xlabel("b value")
plt.ylabel("% of defectors")
plt.title(f"time of evolution: {T_max}") 
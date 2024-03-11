# %%
import imageio
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
# world == 0 -> always cooperate
# world == 1 -> always defect
# world == 2 -> random
# cc (0) = 1
# cd (1) = 0
# dc (10)= b
# dd (11)= 0
# %%


def evolve_one_step(world, b):
    vals = calculate_play_values(world, b)
    # print("ALL:\n", vals)
    evolved_world = choose_new_strat(world, vals)
    return evolved_world


def choose_new_strat(world, vals_world):
    rolls = [(1, 0), (0, 1), (0, 0), (-1, 0), (0, -1),
            (-1, 1), (1, -1), (-1, -1), (1, 1)]
    new_strats = np.copy(world)
    max_vals = np.copy(vals_world)
    print("vals for strats:\n", max_vals)

    for roll_dir in rolls:
        world_rolled = np.roll(world, roll_dir, axis=(0, 1))
        vals_world_rolled = np.roll(vals_world, roll_dir, axis=(0, 1))

        new_strats = np.where(vals_world_rolled > max_vals,
                              world_rolled, new_strats)
        max_vals = np.where(vals_world_rolled > max_vals,
                            vals_world_rolled, max_vals)
    print("new strats:\n",new_strats)
    return new_strats

@jit(nopython = True)
def calculate_play_values(world, b, tft = 2, pav= 3, rnd = 4, M =5):
    L1 = world.shape[0]
    L2 = world.shape[1]
    play_vals = np.zeros((L1,L2), dtype=float)
    for i in range(L1):
        for j in range(L2):
            r = (i+1)%L1
            t = (j-1)%L2
            b = (j+1)%L2
            current_strat = world[i,j]
            # play_self(i,j)
            
            # play(i,j-1)
            # play(i+1,j-1)
            # play(i+1,j)
            # play(i+1,j+1)
@jit(nopython = True)
def play(curr,roll,M):
    for i in range(M):
        if i == 0:
            if curr == 2 or curr == 3:
                curr_choice = 0
            if roll == 2 or roll == 3:
                roll_choice = 0
        else:
            if curr == 2:
                curr_choice = prev_r
            elif curr == 3:
                curr_choice =         
        if curr == 4:
            curr_choice = np.random.randint(0,2)
            prev_c = curr
        if roll == 4:
            roll_choice = np.random.randint(0,2)
            prev_r = roll_choice

@jit(nopython=True)
def play_self(curr,M):
    for i in range(M):
        if curr == 2 or curr = 3:
            curr = 0
            prev = 0


# def calculate_play_values(world, b, tft = 2, pav = 3, rnd = 4, M=5):
#     rolls = [(1, 0), (0, 1), (0, 0), (-1, 0), (0, -1),
#             (-1, 1), (1, -1), (-1, -1), (1, 1)]
#     rolls = np.array(rolls)
#     new_world_vals = np.zeros((world.shape[0], world.shape[1]))

#     for roll_dir in rolls:
#         for i in range(M):
#             world_choice_for_this_game = np.where(world == rnd, np.random.randint(0, 2, (world.shape[0], world.shape[1])), world)
#             if i == 0:
#                 world_choice_for_this_game = np.where(world == tft, 0, world_choice_for_this_game)
#                 world_choice_for_this_game = np.where(world == pav, 0, world_choice_for_this_game)
#                 world_choice_rolled = np.roll(world_choice_for_this_game, roll_dir, axis=(0,1))
                
#                 played = 10*world_choice_for_this_game + world_choice_rolled #receiver|incoming
#                 # print("playing vals:\n", played)
#                 played_vals = b*(played == 10).astype(int) + (played == 0).astype(int)
#                 previous_choice_receiving = np.copy(world_choice_for_this_game)
#                 previous_choice_incoming = np.copy(np.roll(world_choice_for_this_game, roll_dir, axis=(0,1)))
#                 pavlov_receiving_switched_outcomes = np.where(played_vals > 0, world_choice_for_this_game, np.logical_not(world_choice_for_this_game))
#                 pavlov_incoming_switched_outcomes = np.where
#             else:
#                 world_choice_rolled = np.roll(world_choice_for_this_game, roll_dir, axis=(0,1))

#                 receiving = np.where(world_choice_for_this_game == tft, previous_choice_incoming, world_choice_for_this_game)
#                 incoming = np.where(world_choice_rolled == tft, previous_choice_receiving, world_choice_rolled)
#                 receiving = np.where(world_choice_for_this_game == pav, , world_choice_for_this_game)
#                 incoming = np.where(world_choice_rolled == pav, , world_choice_rolled)
                
#                 played = 10*receiving + incoming #receiver|incoming, points only for receiver
#                 played_vals = b*(played == 10).astype(int) + (played == 0).astype(int)
#                 new_world_vals += played_vals
#                 previous_choice =
#     return new_world_vals/M


def show_world(min_val, max_val, num_colors, world, t, filenames, b, options, save=True):
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
    else:
        plt.show()


# %%
# value assignments
L = 5
coop_val = 0
defect_val = 1
random = 4
tft = 2
pavlov = 3
all_options = {"coop":coop_val,
               "defect": defect_val,
                # "random": random,
                 "TitForTat": tft}
                #  "Pavlov": pavlov}
maxval = max(all_options.values())
b = 1.5
T_max = 2
# %%
# generating worlds
world = np.random.randint(0, maxval+1, (L, L))
# world = np.array([[1,2,1],[1,2,1]])
show_world(0, maxval, len(all_options), world, _, [],b, all_options, save= False)

# condition_cooperator = [world!=coop_val, world == coop_val]
# choice_cooperator = [0, coop_val]

# condition_defector = [world!=defect_val, world == defect_val]
# choice_defector = [0, defect_val]

# cooperators_world = np.select(condition_cooperator, choice_cooperator)
# defectors_world = np.select(condition_defector, choice_defector)

# %%
filenames = []
for t in range(T_max):
    show_world(0, maxval, len(all_options), world, t, filenames,b, all_options)
    world = evolve_one_step(world, b)
show_world(0, maxval, len(all_options), world, t+1, filenames,b, all_options)
# %%
with imageio.get_writer('t1.gif', mode='I', fps = 30) as writer:
    for frame in filenames:
        image = imageio.imread(frame)
        writer.append_data(image)
writer.close()
##############################################
# %%
# %%
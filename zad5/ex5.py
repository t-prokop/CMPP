#%%
#define
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
# from scipy.signal import convolve2d

d = 1
s = -1
checkboard = np.array([[d,s,d],[s,d,s],[d,s,d]])

@jit(nopython=True)
def make_world(L):
    return np.random.randint(0, 2, (L, L))

@jit(nopython=True)
def calc_neighbourhood_vals(world, L):
    world = world.ravel()
    neighbour_val = np.roll(world, -L-1) + 2*np.roll(world, -L) + 4*np.roll(world, -L+1) + 8*np.roll(
        world, -1) + 16*world + 32*np.roll(world, -1) + 64*np.roll(world, L-1) + 128*np.roll(world, L) + 256*np.roll(world, L+1)
    return neighbour_val.reshape(L,L)

def evolve_world(world,L, rule):
    return rule[calc_neighbourhood_vals(world, L).astype(int)]

# @jit(nopython=True)
def calc_fitness(world, diag_good_val = 10, side_bad_val = -8, diag_bad_val = -12):
    left_bad_sum = np.sum((world == np.roll(world, 1, axis = 0)))
    top_bad_sum = np.sum((world == np.roll(world, 1, axis = 1)))
    top_right_good = (world == np.roll(world, (-1,1), axis = (0,1)))
    bottom_right_good = (world == np.roll(world, (1,1), axis = (0,1)))
    top_right_bad = np.logical_not(top_right_good)
    bottom_right_bad = np.logical_not(bottom_right_good)
    side_bad_sum = left_bad_sum + top_bad_sum
    diag_good_sum = np.sum(top_right_good) + np.sum(bottom_right_good)
    diag_bad_sum = np.sum(top_right_bad) + np.sum(bottom_right_bad)
    
    fitness = diag_good_val*diag_good_sum + side_bad_val*side_bad_sum + diag_bad_sum*diag_bad_val
    return fitness

def convolve_calc_fitness(world):
    return np.sum(convolve2d(world, checkboard, boundary = 'wrap'))

def reproduce_chromosomes(chromosomes, fitnesses, num_chromosomes, num_surviving = 10):
    chromos_sorted, _ = zip(*reversed(sorted(zip(chromosomes, fitnesses), key = lambda x: x[1])))
    chromos_sorted = list(chromos_sorted[:num_surviving])
    for i in range(5):
        mum = chromos_sorted[0]
        dad = chromos_sorted[1]
        child = np.concatenate((mum[:len(mum)//2],dad[len(dad)//2:]))   
        chromos_sorted.append(child)
    
    for i in range(3):
        mum = chromos_sorted[0]
        dad = chromos_sorted[2]
        child = np.concatenate((mum[:len(mum)//2], dad[len(dad)//2:]))
        chromos_sorted.append(child)
    
    for i in range(2):
        mum = chromos_sorted[3]
        dad = chromos_sorted[4]
        child = np.concatenate((mum[:len(mum)//2], dad[len(dad)//2:]))
        chromos_sorted.append(child)

    mutate_choice = np.random.randint(0,num_chromosomes,2)
    for index in mutate_choice:
        place_mutated = np.random.randint(0, 2**9)
        chromos_sorted[index][place_mutated] = 1 - chromos_sorted[index][place_mutated]
        
    return chromos_sorted
#%%
#init

L = 50
num_chromosomes = 20
#constants in programme
#num_surviving = 10

chromosomes = []
for i in range(num_chromosomes):
    chromosomes.append(np.random.randint(0, 2, 2**9))

w0 = make_world(L)

T_evol = 120
num_generations = 500
worlds_reps = 10
#%%
max_fitnesses = []
avg_fitnesses = []
for nun_gen in tqdm(range(num_generations)):
    chromos_fitnesses = []
    for chromo in chromosomes:
        this_chromo_fit = 0
        for wnum in range(worlds_reps):
            w = make_world(L)
            for t in range(T_evol):
                w = evolve_world(w, L, chromo)
            this_chromo_fit += calc_fitness(w)
        chromos_fitnesses.append(this_chromo_fit)
    max_fitnesses.append(max(chromos_fitnesses))
    avg_fitnesses.append(np.mean(chromos_fitnesses))   
    chromosomes = reproduce_chromosomes(chromosomes, chromos_fitnesses, num_chromosomes)
#%%
#choose best chromosome
chromos_fitnesses_last = []
for chromo in chromosomes:
        this_chromo_fit = 0
        for wnum in range(worlds_reps):
            w = make_world(L)
            for t in range(500):
                w = evolve_world(w, L, chromo)
            this_chromo_fit += calc_fitness(w)
        chromos_fitnesses_last.append(this_chromo_fit)
    
best_chromo = chromosomes[np.argmax(chromos_fitnesses_last)]
#%%
plt.plot(max_fitnesses, label = 'max')
plt.plot(avg_fitnesses, label = 'avg')
plt.xlabel('generation')
plt.ylabel('fitness (positive-definite)')
plt.show()
#%%
filenames = []
w = make_world(L)
for i in range(1000):
    w = evolve_world(w, L, best_chromo)
    plt.imshow(w, cmap = 'gray')
    plt.axis('off')
    plt.savefig(f'./ex5_imgs/frame{i}.png')
    filenames.append(f'./ex5_imgs/frame{i}.png')
    plt.close()
# %%
import imageio
with imageio.get_writer('new_evolution.gif', mode='I') as writer:
    for frame in filenames:
        image = imageio.imread(frame)
        writer.append_data(image)
writer.close()

# %%
chromosomes = []
for i in range(num_chromosomes):
    chromosomes.append(np.load("./best_16k.npy"))
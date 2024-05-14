#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#init
def neighbours(i, N = 100):
    ix, iy = i
    return [(ix,(iy+1)%N), (ix,(iy-1)%N), ((ix+1)%N,iy), ((ix-1)%N,iy)]

def update(i, ising, h_loc, list_to_flip, flipped_set):
    ising[i] = 1
    for j in neighbours(i, N = 100):
        h_loc[j] += 2
        if h_loc[j] > 0 and j not in flipped_set:
            list_to_flip.append(j)
            flipped_set.add(j)
    return
#%%
#TASK 1
R_list = [0.7,0.9, 1.4]
N_rand = 1000
N = 100
mean_avalance_sizes = []
for R in R_list:
    avalanche_sizes = []
    for num_rnd in tqdm(range(N_rand)):
        ising = np.full((N,N), -1)
        flipped = set()
        # h_rnd = np.random.normal(0,R,(N,N))
        h_rnd = np.random.randn(N,N)*R
        h_loc = np.full((N,N), -4) + h_rnd
        spinlist = []
        i_trig = np.unravel_index(np.argmax(h_loc), h_loc.shape)
        H_to_trig = -h_loc[i_trig]
        h_loc += np.full((N,N), H_to_trig)
        spinlist.append(i_trig)
        flipped.add(i_trig)
        while (len(spinlist) > 0):
            i = spinlist.pop(0)
            update(i, ising, h_loc, spinlist, flipped)
        avalanche_sizes.append(len(flipped))
        # plt.imshow(ising)
        # plt.show()
        # plt.clf()
    mean_avalance_sizes.append(np.mean(avalanche_sizes))
    
for i in range(len(R_list)):
    print(f'R = {R_list[i]}, mean avalanche size = {mean_avalance_sizes[i]}')



#%%
N2 = 300
#TASK 2
def update2(i, ising, h_l, list_to_flip, flipped_set, count, counter):
    ising[i] = 1
    for j in neighbours(i, N2):
        h_l[j] += (-2)*ising[i]*ising[j]
        if h_l[j] > 0 and j not in flipped_set:
            count[j] = counter
            list_to_flip.append(j)
            flipped_set.add(j)
    return
'''
def reverse_update2(i, ising, list_to_flip, flipped_set, count, counter):
    ising[i] = -1
    for j in neighbours(i):
        h_l[j] += (-2)*ising[i]*ising[j]
        if h_l[j] < 0 and j not in flipped_set:
            count[j] = counter
            list_to_flip.append(j)
            flipped_set.add(j)
    return
'''
R_list2 = [0.9,1.4,2.1]
for R in R_list2:
    H = -3
    h_rnd2 = np.random.randn(N,N)*R
    count = np.zeros((N,N))
    h_loc2 = np.full((N,N), -4) + h_rnd2
    ising2 = np.full((N,N), -1)
    maglist = []
    H_list = []
    counter = 1
    flipped = set()
    avalanche_sizes = []
    while (H < 3):
        aval_size = 0
        counter += 1
        M = np.sum(ising2)/(N**2)
        maglist.append(M)
        H_list.append(H)
        spinlist = []
        to_trig_energy = np.where(ising2==(-1), h_loc2,-200)
        i_trig = np.unravel_index(np.argmax(np.where(ising2==(-1), h_loc2,-200)), h_loc2.shape)
        H_to_trig = -to_trig_energy[i_trig]
        H += H_to_trig
        # print(H_to_trig, H)
        h_loc2 += np.full((N,N), H_to_trig)
        spinlist.append(i_trig)
        flipped.add(i_trig)
        while (len(spinlist) > 0):
            aval_size += 1
            i = spinlist.pop(0)
            update2(i, ising2, h_loc2, spinlist, flipped, count, counter)
        avalanche_sizes.append(aval_size)
    
        # plt.imshow(ising2)
        # plt.show()
        # plt.clf()
    plt.scatter(H_list, maglist)
    plt.xlabel('H_ext')
    plt.ylabel('M')
    plt.xlim(-3,3)
    plt.title(f'R = {R}')
    plt.show()
    plt.clf()
    plt.imshow(count)
    plt.show()
    plt.clf()
    numbins = 20
    _, bins = np.histogram(avalanche_sizes, bins = numbins)
    logscale_bins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),numbins)
    plt.hist(avalanche_sizes, bins = logscale_bins)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.clf()    
    # while (H > -3):
    #     flipped = set()
    #     spinlist = []
    #     i_trig = np.unravel_index(np.argmax(np.where(ising==(-1), h_loc2,-200)), h_loc2.shape)
    #     H_to_trig = -h_loc2[i_trig]
    #     H -= H_to_trig
    #     h_loc2 += np.full((N,N), H_to_trig)
    #     spinlist.append(i_trig)
    #     flipped.add(i_trig)
    #     while (len(spinlist) > 0):
    #         i = spinlist.pop(0)
    #         reverse_update2(i, ising, spinlist, flipped)
        
    


# %%

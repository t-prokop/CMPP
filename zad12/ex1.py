# %%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
#%%
D = np.array([[1, 1, 1, 1, -1], [-1, 1, -1, -1, 1], [-1, 1, -1, -1, 1], [-1,
             1, -1, -1, 1], [-1, 1, -1, -1, 1], [-1, 1, -1, -1, 1], [-1, 1, 1, 1, -1]])
J = np.array([[1, 1, 1, 1, 1], [-1, -1, -1, 1, -1], [-1, -1, -1, 1, -1],
             [-1, -1, -1, 1, -1], [-1, -1, -1, 1, -1], [1, -1, -1, 1, -1], [1, 1, 1, -1, -1]])
C = np.array([[-1, 1, 1, 1, 1], [1, -1, -1, -1, -1], [1, -1, -1, -1, -1],
             [1, -1, -1, -1, -1], [1, -1, -1, -1, -1], [1, -1, -1, -1, -1], [-1, 1, 1, 1, 1]])
M = np.array([[1, -1, -1, -1, 1], [1, 1, -1, 1, 1], [1, -1, 1, -1, 1],
             [1, -1, -1, -1, 1], [1, -1, -1, -1, 1], [1, -1, -1, -1, 1], [1, -1, -1, -1, 1]])

lshape = D.shape


def show_letters(ltr):
    for letter in ltr:
        plt.imshow(letter, cmap='gray')
        plt.show()


def calc_W_matrix(ltr):
    dim = ltr[0].shape[0]
    W = np.zeros((dim, dim))
    for letter in ltr:
        W += np.outer(letter, letter)

    W /= len(ltr)
    W -= np.eye(dim)

    return W

def evolve(W, pattern, T):
    for _ in range(T):
        ret = np.sign(np.dot(W, pattern))

    return ret


def flip_pattern(pattern, num_flips=5):
    poslist = np.random.choice(len(pattern), num_flips, replace=False)
    for pos in poslist:
        pattern[pos] *= -1

    return pattern


# %%
list_to_remember = [D.flatten(), J.flatten(), C.flatten(), M.flatten()]

Wm = calc_W_matrix(list_to_remember)


def test(letter, s=5):
    probe = letter.flatten()

    probe_final = evolve(Wm, probe, s)

    fig, ax = plt.subplots(2)
    ax[0].imshow(letter)
    ax[1].imshow(probe_final.reshape(letter.shape))
    plt.show()


def recognise(letter, n=5, s=5, show=False, letter_list=None):
    flipped = flip_pattern(letter.flatten(), n)

    final = evolve(Wm, flipped, s)
    if show:
        fig, ax = plt.subplots(4)
        ax[0].imshow(letter)
        ax[0].set_title("original")
        ax[1].imshow(flipped.reshape(letter.shape))
        ax[1].set_title("flipped")
        ax[2].imshow(final.reshape(letter.shape))
        ax[2].set_title("ewolucja")
    if letter_list is not None:
        overlap_list = []
        for l in letter_list:
            overlap_list.append(
                np.dot(final, l.flatten()) / l.flatten().shape[0])
        ax[3].imshow(letter_list[np.argmax(overlap_list)].reshape(lshape))
        ax[3].set_title("max overlap")
        plt.show()
        if (letter == letter_list[np.argmax(overlap_list)].reshape(lshape)).all():
            print("zgadniete")
            return 1
        else:
            print("PORAZKA")
            return 0
    overlap = np.dot(final, letter.flatten()) / letter.flatten().shape[0]
    return overlap


sigma = 5
tau = 5
avg_overlap_list = []
Imin = 5
Imax = 15
for i in range(Imin,Imax):
    avg_overlap = 0
    for t in range(tau):
        randind = np.random.randint(0, len(list_to_remember))
        letter = list_to_remember[randind].reshape(lshape)
        avg_overlap += recognise(letter, n=i, s=sigma, show=True, letter_list = list_to_remember)
    avg_overlap /= tau
    avg_overlap_list.append(avg_overlap)
#%%
plt.plot(range(Imin,Imax), (avg_overlap_list))
plt.show()

# %%
#EXTRA
def gen_patterns(J,N):
    pats = []
    intmax = 2**J
    rand_reprs = np.random.choice(J, N, replace=False)
    for i in range(N):
        pats.append((2*np.array(list(map(int,np.binary_repr(rand_reprs[i], width = J)))) - 1))

    return pats

def gen_W_matrix(J,N,patterns):
    W = np.zeros((J,J))
    for pat in patterns:
        W += np.outer(pat,pat)
    
    W /= N
    W -= np.eye(J)
    return W

@jit(nopython = True)
def evolve(W, pattern):
    ret = np.sign(np.dot(W, pattern))
    return ret


def perturb(patterns):
    cp = [pat.copy() for pat in patterns]
    for pat in cp:
        posflip = np.random.choice(J, numflips, replace=False)
        pat[posflip] *= -1
    
    return cp

def stabilise(patterns, W):
    retlist = []
    for pat in patterns:
        while (evolve(W,pat) != pat).any():
            pat = evolve(W,pat)
        retlist.append(evolve(W,pat))

    return retlist

def calc_overlap(perturblist, stablist, J):
    overlap = 0
    for i in range(N):
        overlap += np.dot(perturblist[i],stablist[i]) / J

    return overlap

def gen_overlap(J,N,numflips):
    patterns = gen_patterns(J,N)
    flipped = perturb(patterns)
    W_matrix = gen_W_matrix(J, N, patterns)
    stab = stabilise(flipped, W_matrix)
    overlap = calc_overlap(flipped, stab, J)

    return overlap/N

#%%
J = 30#pattern len
numflips = 1
# numrec = 1000


overlaplist = []
I = 100
Nmax = 100


for N in range(1,Nmax+1):
    overlaplist.append(gen_overlap(J,N, numflips))
#%%
xaxis = np.array(range(Nmax))/J
plt.plot(xaxis,overlaplist)

#%%
N = 90
gen_overlap(J,N, numflips, numrec)

# pat = gen_patterns(J,N)
# W = gen_W_matrix(J,N,pat)
# flipped = perturb(pat)
# stab = stabilise(flipped, W)
# %%

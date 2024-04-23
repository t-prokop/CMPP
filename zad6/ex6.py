#%%
import numpy as np
import matplotlib.pyplot as plt

def GOE(N):
    h = np.random.randn(N, N)
    return (h + np.conjugate(h.T))/2

def GUE(N):
    h = 1/np.sqrt(2)*(np.random.randn(N, N) + 1j*np.random.randn(N, N))
    return (h + np.conjugate(h.T))/2

def check_eigenvals(h,eigenvals):
    for ev in eigenvals:
        assert (np.linalg.det(h-ev*np.eye(h.shape[0])).round(10)== 0)

def generate_eigenvals(nsample, N, gen_func):
    eigenvalues = []
    for i in range(nsample):
        evs = np.linalg.eigh(gen_func(N))[0]
        for ev in evs:
            eigenvalues.append(ev)
    return eigenvalues

def generate_histo(eigenvalues, N, numbins = 200, name = 'GOE'):
    n, bins, _ = plt.hist(eigenvalues, numbins, density = True)
    plt.xlabel("Eigenvalues")
    plt.ylabel("Density")
    def wigner_semi(E,N):
        R = np.sqrt(2*N)
        return 2/(np.pi*R**2)*np.sqrt(R**2-E**2)
    plt.plot(bins, wigner_semi(bins,N), label = 'Wigner semi-circle')
    plt.legend()
    plt.savefig(f"./hist_evs_{N}_{name}.png")
    plt.show()
#%%
evs = generate_eigenvals(20000,6, GOE)
generate_histo(evs,6)

evs = generate_eigenvals(10000,20, GOE)
generate_histo(evs,20)

evs = generate_eigenvals(500,200, GOE)
generate_histo(evs,200)
#%%
#EX 2
def folded_eigenvals(nsample, N, gen_func):
    ret = []
    for i in range(nsample):
        evs = np.linalg.eigh(gen_func(N))[0]
        if N < 10:
            evs = np.sort(evs)[N//2:N//2 + 2]
            d = np.diff(evs)
            # print(d)
        else:
            evs = np.sort(evs)[N//4:3*N//4]
            d = np.diff(evs)
        for val in d:
            ret.append(val)
    return ret

def func_goe(s):
    return np.pi/2 * s * np.exp(-np.pi/4 * s**2)
def func_gue(s):
    return 32/np.pi**2 * s**2 * np.exp(-4/np.pi * s**2)

def hist_spacings(eigenvals, func, name, N, numbins = 200):
    eigenvals /= np.mean(eigenvals)
    n, bins, _ = plt.hist(eigenvals, numbins, density = True)
    plt.xlabel("EV spacings")
    plt.ylabel("Density")
    plt.title(N)
    plt.plot(bins, list(map(func,bins)), label = 'Wigner spacing function')
    plt.legend()
    plt.savefig(f"./hist_spacings_{N}_{name}.png")
    plt.show()
#%%
evs_gue = folded_eigenvals(10000,8,GUE)
evs_goe = folded_eigenvals(10000,8,GOE)
hist_spacings(evs_gue, func_gue, 'GUE', 8)
hist_spacings(evs_goe, func_goe, 'GOE', 8)
#%%
evs_gue = folded_eigenvals(1000,200, GUE)
evs_goe = folded_eigenvals(1000,200,GOE)
hist_spacings(evs_gue, func_gue, 'GUE', 200)
hist_spacings(evs_goe, func_goe, 'GOE', 200)


#%%
import scipy as sc
#EXTRA
def phi(i,x):
    return np.exp(-(x**2)/2)*sc.special.eval_hermite(i,x)/(np.sqrt((2**i)*np.math.factorial(i)*np.sqrt(np.pi)))
    # return np.exp(-(x**2)/2)*sc.special.eval_hermite(i,x)
def analytic(n,x):
    r = 0
    for i in range(n):
        r += phi(i,x)**2
    return r

def an(n):
    return lambda x: analytic(n,x)/n

def generate_histo(eigenvalues, N, func, numbins = 200, name = 'GOE'):
    n, bins, _ = plt.hist(eigenvalues, numbins, density = True)
    plt.xlabel("Eigenvalues")
    plt.ylabel("Density")
    plt.plot(bins, list(map(func,bins)), label = 'analytic function')
    plt.legend()
    plt.savefig(f"./extra_hist_evs_{N}_{name}.png")
    plt.show()

#%%
evs_gue_extra_1 = generate_eigenvals(50000,1,GUE)
generate_histo(evs_gue_extra_1,1, an(1))
#%%
evs_gue_extra_5 = generate_eigenvals(50000,5,GUE)
generate_histo(evs_gue_extra_5,5, an(5))

#%%
evs_gue_extra_200 = generate_eigenvals(10000,200,GUE)
hist_spacings(evs_gue_extra_200, lambda x:0, 'GUE_extra', 200)
# %%
if 0:
    ham = GOE(2)

    eigenvals = list(np.linalg.eigh(ham)[0])
    check_eigenvals(ham, eigenvals)
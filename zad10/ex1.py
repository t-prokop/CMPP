# %%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm


@jit(nopython=True)
def rf(x0, x1, dt):
    return np.exp(-(x0-x1)**2/(2*dt))

@jit(nopython=True)
def mcs(path, beta, N, delta):
    path_arr = np.copy(path)
    # print(N)
    # print(beta)
    # print(delta)
    dt = beta/N
    acc_cntr = 0
    for xyz in range(N):
        k = np.random.randint(0, N)
        xk = path_arr[k]
        nxk = path_arr[k] + np.random.uniform(-delta, delta)
        xkp = path_arr[(k+1) % N]
        xkm = path_arr[(k-1) % N]
        pi_a = rf(xkm, xk, dt)*rf(xk, xkp, dt)*np.exp(-(dt*xk**2)/2)
        pi_b = rf(xkm, nxk, dt)*rf(nxk, xkp, dt)*np.exp(-(dt*nxk**2)/2)
        gamma = pi_b/pi_a
        if (np.random.uniform(0, 1) < gamma):
            path_arr[k] = nxk
            acc_cntr += 1
    return path_arr, np.mean(path_arr), np.var(path_arr), acc_cntr/N, path_arr[0], path_arr[N//2]


# %%
# TASK 1
beta = 4

delta = 1
N = 8
num_steps = 10000
path = np.random.uniform(-delta, delta, N)
# %%
x_mean_arr = []
x_var_arr = []
for i in tqdm(range(num_steps)):
    path, xm, xv, _, _, _ = mcs(path, beta, N, delta)
    x_mean_arr.append(xm)
    x_var_arr.append(xv)
# %%
plt.scatter(range(num_steps), x_mean_arr, s=0.1, label='x mean', c='g')
plt.scatter(range(num_steps), x_var_arr, s=0.1, label='x var', c='r')
plt.hlines(np.mean(x_mean_arr), 0, num_steps,
           'g', '-', label='mcs mean of mean')
plt.hlines(np.mean(x_var_arr), 0, num_steps, 'r', '-', label='mcs mean of var')
plt.legend()
plt.show()
print(
    f'mcs-mean of mean: {np.mean(x_mean_arr)}, mcs-mean of var: {np.mean(x_var_arr)}')
# %%
# TASK 2
num_trials = 100

delta = 1
N = 8
beta = 4
num_steps = 10000
path = np.random.uniform(-delta, delta, N)

r_acc_list = []
delta_list = []
for i in range(num_trials):
    acc_tot = 0
    for j in range(num_steps):
        path, _, _, r_acc, _, _ = mcs(path, beta, N, delta)
        acc_tot += r_acc
    mean_r_acc = acc_tot/num_steps
    delta *= mean_r_acc/0.75
    delta_list.append(delta)
    r_acc_list.append(mean_r_acc)

plt.plot(delta_list, label='delta')
plt.plot(r_acc_list, label='acceptance rate')
plt.legend()
plt.show()
delta = np.mean(delta_list[:int(0.8*num_trials)])
print(f'Final delta proposition: {delta}')
# %%
N = 40
beta = 10
num_steps = 100000
path = np.random.uniform(-delta, delta, N)
print(len(path), N, beta, delta)
# %%
x_mean_arr = []
x_var_arr = []
x0_arr = []
xh_arr = []
for i in range(num_steps):
    path, xm, xv, acc, x0, xh = mcs(path, beta, N, delta)
    # print(acc)
    x_mean_arr.append(xm)
    x_var_arr.append(xv)
    x0_arr.append(x0)
    xh_arr.append(xh)
# %%
# plt.scatter(range(num_steps), x_mean_arr, s = 0.1, label = 'x mean', c = 'g')
# plt.scatter(range(num_steps), x_var_arr, s = 0.1, label = 'x var', c = 'r')
# plt.hlines(np.mean(x_mean_arr), 0, num_steps, 'g', '-', label='mcs mean of mean')
# plt.hlines(np.mean(x_var_arr), 0, num_steps, 'r', '-', label='mcs mean of var')
# plt.legend()
# plt.show()
# print(f'mcs-mean of mean: {np.mean(x_mean_arr)}, mcs-mean of var: {np.mean(x_var_arr)}')
# %%
print(np.var(x0_arr), np.var(xh_arr))


def sigma(b):
    return 1/(2*np.tanh(b/2))


def pi_b(x, b):
    sig_sq = sigma(b)
    return np.exp(-x**2/(2*sig_sq))/np.sqrt(2*np.pi*sig_sq)


def pi(x): return pi_b(x, beta)


def pi2(x):
    return np.sqrt(np.tanh(beta/2))*np.exp(-np.tanh(beta/2)*x**2)/np.sqrt(np.pi)


x_axis = np.linspace(-3, 3, 10000)
numbins = 100

plt.hist(x0_arr, bins=numbins, label='x0', density=True)
plt.plot(x_axis, list(map(pi2, x_axis)), c='black')
plt.legend()
plt.show()

plt.hist(xh_arr, bins=numbins, label='xh', density=True)
plt.plot(x_axis, list(map(pi2, x_axis)), c='black')
plt.legend()
plt.show()

# %%

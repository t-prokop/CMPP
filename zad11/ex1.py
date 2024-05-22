# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm
from scipy.linalg import eigh

sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])
I = np.eye(2)


def Sxyz(i, n, s_matrix): #recursive definition of Sx_i
    if i == 1:
        ret = np.copy(s_matrix)
    else:
        ret = np.copy(I)
    k = 1
    while(k < n):
        if k+1 == i:
            ret = np.kron(s_matrix, ret)
        else:
            ret = np.kron(I, ret)
        k += 1
    return ret


def H0(n, s_matrix = sx):
    ret = np.zeros((2**n,2**n))
    for i in range(1,n+1):
        ret += Sxyz(i,n, s_matrix)
    return -ret


def H1(n, j_arr, h_arr, pairs, s_matrix = sz):
    ret = np.zeros((2**n, 2**n))
    for j,pair in enumerate(pairs):
        ret += j_arr[j]*np.dot(Sxyz(pair[0],n,s_matrix),Sxyz(pair[1],n,s_matrix))
    for j in range(1,n+1):
        ret += h_arr[j-1]*Sxyz(j,n,s_matrix)
    return -ret

def H_tot(l, n, j_arr, h_arr, pairs, s0_matrix = sx ,s1_matrix = sz):
    return (1-l)*H0(n, s0_matrix) + l*H1(n, j_arr, h_arr, pairs, s1_matrix)

# print(np.kron(I, np.kron(I, sx)))
# print(Sxyz(1,3, sx))
# print((Sx(1,3) == np.kron(I, np.kron(I, sx))).all())

# print(H0(3))
# print(-1*( np.kron(I,np.kron(I,sx)) + np.kron(I,np.kron(sx,I)) + np.kron(sx,np.kron(I,I)) ))
# print((H0(3) == -1*( np.kron(I,np.kron(I,sx)) + np.kron(I,np.kron(sx,I)) + np.kron(sx,np.kron(I,I)))).all())

pairs = [(1,2), (2,3), (3,1)]
J_arr = [-0.4,-1.6,-1.0]
h_arr = [-0.5, 0.5, -0.1]
N = 3
# H(0,N,J_arr,h_arr,pairs)
# %%
#TASK 1
l_arr = np.linspace(0,1,1000)

pairs = [(1,2), (2,3), (3,1)]
J_arr = [-0.4,-1.0,-1.6]
h_arr = [-0.5, 0.5, -0.1]
N = 3

def groundstate(l):
    en,u = eigh(H_tot(l,N,J_arr,h_arr,pairs))
    return en[0], u[:,0]

evals = []
evecs = []
s_exp_1 = []
s_exp_2 = []
s_exp_3 = []
for l in l_arr:
    E,V = groundstate(l)
    # assert np.dot(V.conj().T, V).round(5) == 1
    s1, s2, s3 = Sxyz(1,N,sz), Sxyz(2,N,sz), Sxyz(3,N,sz)
    s_exp_1.append(np.dot(V.conj().T, np.dot(s1, V)))
    s_exp_2.append(np.dot(V.conj().T, np.dot(s2, V)))
    s_exp_3.append(np.dot(V.conj().T, np.dot(s3, V)))
    evals.append(E)
    evecs.append(V)
#%%
plt.plot(l_arr, evals)
plt.show()
plt.plot(l_arr, s_exp_1, label = 's1')
plt.plot(l_arr, s_exp_2, label = 's2')
plt.plot(l_arr, s_exp_3, label = 's3')
plt.legend()
plt.show()
#%%
#TASK 2
def gen_democratic(N):
    return np.ones(2**N)/2**(N/2)

H = lambda l:H_tot(l,N,J_arr,h_arr,pairs)

Tm = 2000
M = 20000

dt = Tm/M
tk_arr = np.linspace(0,Tm,M+1)
lk_arr = tk_arr/Tm
psi = gen_democratic(N)
psi_list = []
for i,l_val in enumerate(lk_arr):
    psi = np.dot(expm((-1j)*H(l_val)*dt), psi)
    psi_list.append(psi)
    
    # E,U = eigh(H(l_val))
    # psi = U@np.diag(np.exp((-1j)*E*dt))@U.conj().T@psi
    # psi_list.append(psi)
    
#%%
s1_list = []
s2_list = []
s3_list = []
for psi in psi_list:
    s1_list.append(np.dot(psi.conj().T, np.dot(s1, psi)))
    s2_list.append(np.dot(psi.conj().T, np.dot(s2, psi)))
    s3_list.append(np.dot(psi.conj().T, np.dot(s3, psi)))
# %%
plt.plot(tk_arr, s1_list, label = 's1')
plt.plot(tk_arr, s2_list, label = 's2')
plt.plot(tk_arr, s3_list, label = 's3')
plt.legend()
plt.show()
# %%
succ_prob = []
pf = psi_list[-1]
pf_h = pf.conj().T
for psi in psi_list:
    succ_prob.append(np.abs(np.dot(pf_h, psi))**2)
#%%
plt.plot(tk_arr, succ_prob)
plt.hlines(0.9,0,Tm, color = 'red', label = "prob 0.9")
plt.show()
# %%

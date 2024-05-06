#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
cos = lambda x:np.cos(x)
sin = lambda x:np.sin(x)
exp = lambda x:np.exp(x)
#%%
#task 1
measurements = []
psi0 = np.array([1,0])
a = float(input("Give alpha: "))
assert isinstance(a,float), "Alpha must be a number"

U = np.array([[cos(a/2), sin(a/2)],[-sin(a/2), cos(a/2)]])
psi = np.dot(U, psi0)
print(psi)
for i in range(10000):
    toss = np.random.choice([0,1],p = [np.abs(psi[0])**2 , np.abs(psi[1])**2])
    measurements.append(toss)

plt.bar([0,1], [measurements.count(0)/len(measurements), measurements.count(1)/len(measurements)])
plt.scatter([0,1], [np.abs(cos(a/2))**2, np.abs(sin(a/2))**2], color = 'red')
#%%
#task 2 
I = np.array([[1,0],[0,1]])
D = np.array([[0,1],[-1,0]])
Q = np.array([[1j,0],[0,-1j]])

dd = np.kron(D,D)
dq = np.kron(D,Q)
qd = np.kron(Q,D)
qq = np.kron(Q,Q)

def U(p, t):
    return np.array([[exp(1j*p)*cos(t/2), sin(t/2)],\
        [-sin(t/2),exp(-1j*p)*cos(t/2)]])

def J_JH(gamma):
    return sc.linalg.expm(-1j*gamma*dd/2), sc.linalg.expm(1j*gamma*dd/2)


def assert_norm(state):
    norm = np.linalg.norm(state)
    assert round(norm,2) == 1, f"State must be normalized {norm}; {state}"

def payoff(state, r = 3, s = 0, t = 5, p = 1):
#ret: A payoff, B payoff
    assert_norm(state)
    # print("state", state)
    # print(r*np.abs(state[0])**2, s*np.abs(state[1])**2, t*np.abs(state[2])**2, p*np.abs(state[3])**2)
    return r*np.abs(state[0])**2 + s*np.abs(state[1])**2 + t*np.abs(state[2])**2 + p*np.abs(state[3])**2, r*np.abs(state[0])**2 + t*np.abs(state[1])**2 + s*np.abs(state[2])**2 + p*np.abs(state[3])**2

#%%
#task 2a
strats = [dd, dq, qd, qq]
stratnames = ["dd", "dq", "qd", "qq"]
for i in range(len(strats)):
    player_payoffs = []
    # print(stratnames[i])
    for gamma in np.linspace(0,np.pi/2,200):
        init_state = np.array([1,0,0,0])
        # print(gamma)
        J, Jh = J_JH(gamma)
        assert np.allclose(np.dot(J,Jh), np.kron(I,I)), "J and Jh must be conjugate"
        # one = np.dot(J,init_state)
        # # print("1")
        # assert_norm(one)
        # two = np.dot(strats[i], one)
        # # print("2")
        # assert_norm(two)
        # final_state = np.dot(Jh,two)
        # # print("f")
        # assert_norm(final_state)
        final_state = np.dot(Jh,\
            np.dot(strats[i],\
            np.dot(J,init_state)))
        
        player_payoffs.append(payoff(final_state)[0])
    plt.plot(np.linspace(0,np.pi/2,200), player_payoffs, label = stratnames[i])

plt.legend()

#%%
#task 2b
gamma = np.pi/2
J,Jh = J_JH(gamma)
M = 1/np.sqrt(2) * np.array([[1j,1],[-1,-1j]])

def U_bob(theta):
    return U(0,theta)

a_strats = [I,D,M]
a_stratnames = ["I","D","M"]

for i in range(len(a_strats)):
    a_val_list = []
    for theta in np.linspace(0, np.pi, 1000):
        strat = np.kron(a_strats[i],U_bob(theta))
        
        final_state = np.dot(Jh,\
            np.dot(strat,\
            np.dot(J,init_state)))

        a_val_list.append(payoff(final_state)[0])
    plt.plot(np.linspace(0,np.pi,1000), a_val_list, label = a_stratnames[i])

plt.legend()

#%%
#task 2c

def calc_payoff_A(angles, gamma, U_bob):
    theta, phi = angles
    J,Jh = J_JH(gamma)
    strat = np.kron(U(phi,theta),U_bob)
    final_state = np.dot(Jh,\
            np.dot(strat,\
            np.dot(J,init_state)))

    pf = payoff(final_state)[0] 
    return pf

#%%
from tqdm import tqdm
vals = []
for gamma in tqdm(np.linspace(0,np.pi/2, 1000)):
    pf_a = lambda ang: (-1)*min(calc_payoff_A(ang, gamma, I), calc_payoff_A(ang, gamma, D))
    res = sc.optimize.differential_evolution(pf_a, bounds = [(0,np.pi), (0,np.pi/2)])
    vals.append(-pf_a(res.x))

#%%
# vals = np.array(vals)
# vals = (-1)*vals
plt.plot(np.linspace(0,np.pi/2,1000), vals)
plt.show()
plt.plot(np.linspace(0,np.pi/2, 1000)[1:],np.diff(vals))
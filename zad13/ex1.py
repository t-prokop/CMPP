# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm


def gen_graph(wm, sa):
    g = nx.Graph()
    for i in range(NPPL):
        g.add_node(i, state=sa[i])
    for i in range(NPPL):
        for j in range(i+1, NPPL):
            g.add_edge(i, j, weight=wm[i, j])

    return g


def draw_from_graph(g):  # g - graph
    nx.draw_spring(g, cmap=cm.cool, vmin=0, vmax=1, with_labels=True, node_color=list(nx.get_node_attributes(g, "state").values()), edge_cmap=cm.binary, edge_vmin=0,
                   edge_vmax=1, edge_color=list(nx.get_edge_attributes(g, "weight").values()))


def draw(wm, sa):
    draw_from_graph(gen_graph(wm, sa))


def f_sa(sa):
    return (np.abs(np.array(list(sa)*NPPL).reshape(NPPL, NPPL) - np.array(list(sa)*NPPL).reshape(NPPL, NPPL).T) - 0.25)**3


def evolve_state(wm, sa, onemaster_index, zeromaster_index):
    newstate = np.copy(sa)
    newstate += dt * D * (np.dot(wm, sa) - sa*np.sum(wm, axis=1))
    # newstate[i] = sa[i] + dt * (D * np.sum(wm[i] * (sa - sa[i])) - B * f(sa[i]))
    newstate[onemaster_index] = 1
    newstate[zeromaster_index] = 0
    return newstate


def evolve_weight(wm, sa):
    newweight = np.copy(wm)
    newweight -= B * dt * wm * (1-wm) * f_sa(sa)
    return newweight


def opos_str(wm, sa):
    res = np.sum(np.outer(sa > 0.5, sa <= 0.5) * wm)
    print(res)
    return res


# %%
# init vals
NPPL = 34
edge_list = np.loadtxt("zachary_edge_list.txt")
edge_list = edge_list.astype(int)

w_matrix = np.zeros((NPPL, NPPL))

for edge in edge_list:
    w_matrix[edge[0], edge[1]] = 0.5
    w_matrix[edge[1], edge[0]] = 0.5

state_arr = np.full(NPPL, 0.5)
state_arr[0] = 1
state_arr[33] = 0

assert (w_matrix == w_matrix.T).all()

# evolution
dt = 0.01
D = 5
B = 10
# %%
T_max = 8000

newstate_arr = np.zeros(NPPL)
wm_new = np.zeros((NPPL, NPPL))
opposite_conn_str_list = []
for t in range(T_max):
    opposite_conn_str_list.append(opos_str(w_matrix, state_arr))
    newstate_arr = evolve_state(w_matrix, state_arr)
    wm_new = evolve_weight(w_matrix, state_arr)
    state_arr = np.copy(newstate_arr)
    w_matrix = np.copy(wm_new)

# %%
draw(w_matrix, state_arr)
plt.show()
plt.plot(opposite_conn_str_list)
plt.show()
# %%
# EXTRA
wextra = np.loadtxt('extra_wmatrix.txt')
w_matrix = np.copy(wextra)/np.max(wextra)

state_arr = np.full(NPPL, 0.5)
state_arr[0] = 1
state_arr[33] = 0

assert (w_matrix == w_matrix.T).all()

dt = 0.01
D = 5
B = 10

# %%
T_max = 2200

newstate_arr = np.zeros(NPPL)
wm_new = np.zeros((NPPL, NPPL))
opposite_conn_str_list = []
for t in range(T_max):
    opposite_conn_str_list.append(opos_str(w_matrix, state_arr))
    newstate_arr = evolve_state(w_matrix, state_arr)
    wm_new = evolve_weight(w_matrix, state_arr)
    state_arr = np.copy(newstate_arr)
    w_matrix = np.copy(wm_new)

# %%
draw(w_matrix, state_arr)
plt.show()
plt.plot(opposite_conn_str_list)
plt.show()

# %%
# ZABAWA
NPPL = 10000

dt = 0.01
D = 5
B = 10

state_arr = np.full(NPPL, 0.5)

# WYBRAC NODE'Y Z NAJWIEKSZA LICZBA PLACZEN na masterow
def gen_rand_w_matrix(NPPL, num_hubs = 2):
    assert num_hubs < NPPL
    G = nx.scale_free_graph(NPPL).to_undirected()
    leading_degree = sorted(G.degree, key = lambda it:it[1])[-num_hubs:]
    zeroind, oneind = leading_degree[0][0], leading_degree[1][0]
    G.remove_edge(zeroind, oneind)
    A = nx.to_numpy_array(G)
    A = A/np.max(A)
    return G,A, zeroind, oneind


initgraph, wm, zmi, omi = gen_rand_w_matrix(NPPL)
state_arr[zmi] = 0
state_arr[omi] = 1

# draw(wm, state_arr)
# plt.show()
# %%
T_max = 2500

newstate_arr = np.zeros(NPPL)
wm_new = np.zeros((NPPL, NPPL))
opposite_conn_str_list = []
for t in range(T_max):
    opposite_conn_str_list.append(opos_str(wm, state_arr))
    newstate_arr = evolve_state(wm, state_arr, omi, zmi)
    wm_new = evolve_weight(wm, state_arr)
    state_arr = np.copy(newstate_arr)
    wm = np.copy(wm_new)
# %%
# draw(wm, state_arr)
# plt.show()
plt.plot(opposite_conn_str_list)
plt.show()

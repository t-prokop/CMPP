# %%
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# constants
# vel_vects = [[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]
# vel_vects_reverse = list((-1)*np.array(vel_vects))
# W = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]

# initialization


def calc_f_eq(rho, u, Nx, Ny):
    vel_vects = [[0, 0], [1, 0], [0, 1], [-1, 0],
                 [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    W = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]

    ret_f = np.zeros((9, Nx, Ny))
    norm_u = np.einsum('ijk,ijk->ij', u, u)

    for i in range(9):
        dot = np.einsum('ijk,k->ij', u, vel_vects[i])
        ret_f[i] = W[i]*rho*(1 + 3*dot + 9/2*dot**2 - 3/2*norm_u)

    return ret_f


def calc_f_x0(rho, ux0, Ny):  # calc step 1 f_i^eq
    W = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
    vel_vects = [[0, 0], [1, 0], [0, 1], [-1, 0],
                 [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]

    ret_f = np.zeros((9, Ny))
    norm_ux0 = np.einsum('ij,ij->i', ux0, ux0)

    for i in range(9):
        dot = np.einsum('ij,j->i', ux0, vel_vects[i])
        ret_f[i] = W[i]*rho*(1 + 3*dot + 9/2*dot**2 - 3/2*norm_ux0)

    return ret_f
# ALGORITHM


def boundary_calc(f, u0, Ny, Nx):
    # step 1
    rho_x0 = (2*(f[3, 0, :] + f[6, 0, :] + f[7, 0, :]) + f[0, 0, :] +
              f[2, 0, :] + f[4, 0, :])/(1-np.sqrt(np.sum(u0[0, :, :]*u0[0, :, :], axis=1)))
    u_x0 = u0[0, :, :]
    f_eq_x0 = calc_f_x0(rho_x0, u_x0, Ny)

    # step 2
    f[1, 0, :] = f_eq_x0[1, :]
    f[5, 0, :] = f_eq_x0[5, :]
    f[8, 0, :] = f_eq_x0[8, :]
    f[3, Nx-1, :] = f[3, Nx-2, :]
    f[6, Nx-1, :] = f[6, Nx-2, :]
    f[7, Nx-1, :] = f[7, Nx-2, :]

    return f


def recalc_density(f, Nx, Ny):  # step 3
    e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
         [1, 1], [-1, 1], [-1, -1], [1, -1]]
    rho = np.sum(f, axis=0)
    inv_rho = 1/rho
    u = np.einsum('ijk,il -> jkl', f, e)[:, :, :]*inv_rho[:, :, np.newaxis]
    f_eq = calc_f_eq(rho, u, Nx, Ny)
    return f_eq


def collision(f, f_eq, tau, Nx, Ny, obstacle):  # step 4 and 5
    reverse_i = [0, 3, 4, 1, 2, 7, 8, 5, 6]

    inv_tau = 1/tau
    f_col = (1-inv_tau)*f + inv_tau*f_eq

    # step 5
    rev_f = np.zeros((9, Nx, Ny))
    for i in range(9):
        rev_f[i] = f[reverse_i[i], :, :]

    f_col = np.where(obstacle, rev_f, f_col)
    return f_col


def streaming(f, Nx, Ny):
    vel_vects = [[0, 0], [1, 0], [0, 1], [-1, 0],
                 [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    f_stream = np.zeros((9, Nx, Ny))
    for i in range(9):
        f_stream[i, :, :] = np.roll(f[i, :, :], vel_vects[i], axis=(0, 1))
    return f_stream


def calc_norm_u_from_f(f):
    e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
         [1, 1], [-1, 1], [-1, -1], [1, -1]]
    rho = np.sum(f, axis=0)
    inv_rho = 1/rho
    u = np.einsum('ijk,il -> jkl', f, e)[:, :, :]*inv_rho[:, :, np.newaxis]
    norm_u = np.einsum('ijk,ijk->ij', u, u)
    return norm_u


def plot(f, t, fnames, re, save=False, show=False):
    plt.imshow(calc_norm_u_from_f(f), cmap='hot')
    plt.title(f't:{t}, Re:{re}')
    if save:
        plt.savefig(f'./imgs/lbm_{t}.png')
        fnames.append(f'./imgs/lbm_{t}.png')
    if show:
        plt.show()

# %%
def run(Nx, Ny, tau, obstacle, u0, rho0, T_max, Re, fnames=[], save=False):
    # initialize with f_eq for given u0, rho0
    f = calc_f_eq(rho0, u0, Nx, Ny)
    # main loop
    for i in tqdm(range(T_max)):
        f = boundary_calc(f, u0, Ny, Nx)

        f_eq = recalc_density(f, Nx, Ny)

        f = collision(f, f_eq, tau, Nx, Ny, obstacle)

        f = streaming(f, Nx, Ny)

        if i % 100 == 0:
            plot(f, i, fnames, Re, save)
###COPY END
#%%
# %%
# TODO Re = 110
Nx = 520
Ny = 180

diameter = 60

center_x = Nx/4
center_y = Ny/2
assert diameter < Nx/4
cylinder_obj = np.fromfunction(lambda i, j: (i-center_x)**2 + (j-center_y)**2 <= (diameter/2)**2 ,(Nx, Ny))


u_in = 0.04

Re = 10
cross_section = diameter
visc = u_in * Ny / (Re*cross_section)
tau = 3*visc + 0.5


eps = 0.0001
u0 = np.fromfunction(lambda i, j, k: u_in *
                     (1 + eps*np.sin(2*np.pi*j/(Ny-1)))*(k == 0), (Nx, Ny, 2))
rho0 = np.full((Nx, Ny), 1)
plt.imshow(cylinder_obj)
plt.show()
# %%
filenames = []
run(Nx, Ny, tau, cylinder_obj, u0, rho0, 20000, Re, filenames, save=True)
# %%
with imageio.get_writer('./cylinder_re10.gif', mode='I', duration=40) as writer:
    for frame in filenames:
        image = imageio.imread(frame)
        writer.append_data(image)
writer.close()

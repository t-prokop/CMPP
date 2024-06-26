#%%
import numpy as np

id = 449528

c = (np.sqrt(5) -1)/2
s = 10
n= 3
print(f"seria: {s}, c= {c}, nr indeksu: {id}, n = {n} zadan w serii")
#%%
p = ((s + int(n * (id*c - int(id*c)) ) ) % n ) + 1
print(f"zadanie: {p}")
# p = MOD[s+ FLOOR[n*FRAC(id* c)],n]+1
#%%
# %%
phi = np.pi
s2 = np.sqrt(2)
e = np.exp
psip = np.array([1/s2, e(1j*phi)/2, e(1j*phi)/2, 0])

rho = np.outer(psip,psip).reshape(2,2,2,2)
np.round(np.trace(rho, 0,axis1=0,axis2=2),5)
# %%
def partial_trace(rho, dim1, dim2, space, offs= 0, input_assert_prec = 6, output_assert_prec = 6):
    assert(np.round(np.trace(rho), input_assert_prec) == 1)
    a = np.reshape(rho, (dim1,dim2,dim1,dim2))
    if space == 1:
        traxis = (0,2)
    elif space == 2:
        traxis = (1,3)
    else:
        raise(Exception("Bad space indicator"))
    ret = np.trace(a, offs, *traxis)
    assert(np.round(np.trace(ret),output_assert_prec) == 1)
    return ret


v = np.array([1/np.sqrt(14),2/np.sqrt(14), 3/np.sqrt(14),0,0,0])
r = np.outer(v,v)
p = partial_trace(r,2,3,1)
print(np.round(p,3))

v1 = np.array([1/np.sqrt(14),2/np.sqrt(14), 3/np.sqrt(14)])
v2 = np.array([1,0])
print(np.round(v,3),np.round(np.kron(v1,v2),3))
print(np.round(np.outer(v1,v1),3))
# %%
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,1j],[-1j,0]])
sz = np.array([[1,0],[0,-1]])
rho_3_10 = np.eye(4)/4 + np.kron(sx,sx)/16 + np.kron(sy,sy)/16 + np.kron(sz,sz)/8
print(np.trace(rho_3_10))
print(np.trace(rho_3_10@rho_3_10))
pp = np.array([1,0,0,0])
pp_proj = np.outer(pp,pp)
print(np.trace(rho_3_10@pp_proj))
dd = np.array([0,0,0,1])
dd_proj = np.outer(dd,dd)
print(np.trace(dd_proj@rho_3_10))
trip = np.array([0,1/np.sqrt(2), 1/np.sqrt(2),0])
trip_proj = np.outer(trip,trip)
print(np.trace(trip_proj@rho_3_10))
sing = np.array([0,1/np.sqrt(2), -1/np.sqrt(2),0])
sing_proj = np.outer(sing,sing)
print(np.trace(sing_proj@rho_3_10))
# %%
i = np.eye(2)
j2 = 1/4*(np.kron(sx@sx,i) + np.kron(sy@sy,i) + np.kron(sz@sz,i) +\
     2*(np.kron(sx,sx) + np.kron(sy,sy) + np.kron(sz,sz)) +\
     np.kron(i,sx@sx) + np.kron(i,sy@sy) + np.kron(i,sz@sz))
print(np.linalg.eigh(j2)) #linalg się rozjebał? XD
print(np.trace(j2@rho_3_10))
# %%

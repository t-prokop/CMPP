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

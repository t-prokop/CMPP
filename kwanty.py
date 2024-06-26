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

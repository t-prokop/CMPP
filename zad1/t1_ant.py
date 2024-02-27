#%%
import numpy as np
import matplotlib.pyplot as plt
# %%
#TASK 1
#def
class ant:
    def __init__(self,L,i):
        self.pos = np.random.randint(0,L,2)
        self.dir = np.array((-1,0))
        self.L = L
        self.i = i+1

    def rotate_right(self):
        if self.dir[0] == -1:
            self.dir = np.array([0,1])
        elif self.dir[0] == 1:
            self.dir = np.array([0,-1])
        elif self.dir[0] == 0:
            if self.dir[1] == 1:
                self.dir = np.array([1,0])
            elif self.dir[1] == -1:
                self.dir = np.array([-1,0])
        else:
            raise Exception("blad obrotu w prawo")
        
    def rotate_left(self):
        if self.dir[0] == -1:
            self.dir = np.array([0,-1])
        elif self.dir[0] == 1:
            self.dir = np.array([0,1])
        elif self.dir[0] == 0:
            if self.dir[1] == 1:
                self.dir = np.array([-1,0])
            elif self.dir[1] == -1:
                self.dir = np.array([1,0])
        else:
            raise Exception("blad obrotu w lewo")

    def move(self):
            self.pos = (self.pos + self.dir) % L
#%%
#init
L = 1000
num_steps = 1000000
w = np.zeros((L,L), dtype=np.uint8)

num_ants = 4
ants_list = []
for i in range(num_ants):
 a = ant(L,i)
 ants_list.append(a)

print(L)
#%%
#propagate
for t in range(num_steps):
    for a in ants_list:
        if w[a.pos[0], a.pos[1]] == 0:
            w[a.pos[0], a.pos[1]] = a.i
            a.rotate_right()
            a.move()
        else:
            w[a.pos[0], a.pos[1]] = 0
            a.rotate_left()
            a.move()
#%%
plt.imshow(w, interpolation='none', cmap='gray', vmin=0, vmax = num_ants)
plt.show()


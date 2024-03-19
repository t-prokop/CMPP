# %%
import imageio
import numpy as np
import matplotlib.pyplot as plt
import numba

angle_list = np.arange(0, 360, 45)


def p_func(x, h=3, a=2.5, b=4):
    # h = 1, a = 2, b = 5
    return (h+b*x)**a


class ant:
    def __init__(self, init_direction, position, world_shape):
        self.angle = init_direction
        self.pos = np.array(position)
        self.time = 0
        self.shape = world_shape
        self.mode = 1  # 1 - foraging, 2 - returning
        self.sniffing = 2
        self.pheromone_strenght = 1

    def move(self, angle):
        if angle == 0:
            move = np.array([0, 1])
        elif angle == 45:
            move = np.array([1, 1])
        elif angle == 90:
            move = np.array([1, 0])
        elif angle == 135:
            move = np.array([1, -1])
        elif angle == 180:
            move = np.array([0, -1])
        elif angle == 225:
            move = np.array([-1, -1])
        elif angle == 270:
            move = np.array([-1, 0])
        elif angle == 315:
            move = np.array([-1, 1])
        else:
            print(self.angle)
            raise Exception("wrong angle")
        return np.remainder(self.pos + move, self.shape)

    def leave_pheromone(self, world):
        world[self.pos[0], self.pos[1], self.mode] += self.pheromone_strenght

    def switch_mode(self):
        if self.mode == 1:
            self.mode = 2
            self.sniffing = 1
        elif self.mode == 2:
            self.mode = 1
            self.sniffing = 2
        else:
            raise Exception("zly mode")

    def rotate_right(self):
        return (self.angle+45) % 360

    def rotate_left(self):
        return (self.angle-45) % 360

    def turn_around(self):
        self.angle = (self.angle + 180) % 360

    def make_move(self, world, food_counter, beta=0.99, full_str_pheromone_time=50):
        self.leave_pheromone(world)

        fm = self.move(self.angle)
        lm = self.move(self.rotate_left())
        rm = self.move(self.rotate_right())

        if world[fm[0], fm[1], 0] == self.mode:
            # print(self.mode)
            if self.mode == 2:
                food_counter += 1
            # print(f"mode swtiched when: {self.mode}")
            self.pos = fm
            self.time = 0
            self.switch_mode()
            self.turn_around()

        elif world[lm[0], lm[1], 0] == self.mode:
            # print(self.mode)
            if self.mode == 2:
                # print("FOOD COLLECTED!!!")
                food_counter += 1
            # print(f"mode swtiched when: {self.mode}")
            self.pos = lm
            self.time = 0
            self.switch_mode()
            self.angle = self.rotate_left()
            self.turn_around()

        elif world[rm[0], rm[1], 0] == self.mode:
            # print(self.mode)
            if self.mode == 2:
                food_counter += 1
            # print(f"mode swtiched when: {self.mode}")
            self.pos = rm
            self.time = 0
            self.switch_mode()
            self.angle = self.rotate_right()
            self.turn_around()

        else:
            # print("forward",np.remainder(self.pos + self.dir, self.shape),sniffing)
            # print("right",np.remainder(self.pos + self.rotate_right(), self.shape),sniffing)
            # print("left",np.remainder(self.pos + self.rotate_left(), self.shape),sniffing)
            forward_pheromone = world[fm[0], fm[1], self.sniffing]
            right_pheromone = world[rm[0], rm[1], self.sniffing]
            left_pheromone = world[lm[0], lm[1], self.sniffing]

            moves = [fm, rm, lm]
            pheromones = [forward_pheromone, right_pheromone, left_pheromone]
            # print(moves)
            # print(pheromones)

            pheromones = [pheromones[i] for i in range(
                3) if world[moves[i][0], moves[i][1], 0] != 4]
            moves = [moves[i]
                     for i in range(3) if world[moves[i][0], moves[i][1], 0] != 4]
            # print(moves)
            # print(pheromones)
            if len(moves) > 0:
                move_probs = np.array(list(map(p_func, pheromones)))
                move_probs = move_probs/np.sum(move_probs)
                # print(move_probs)
                # if len(pheromones) < 3:
                #     print(f"blocked at: {self.pos}")
                angles = [self.angle, self.rotate_right(), self.rotate_left()]
                # print("move_probs", move_probs)
                move_choice = np.random.choice(range(len(moves)), p=move_probs)
                self.pos = moves[move_choice]
                assert world[self.pos[0], self.pos[1], 0] != 4
                self.angle = angles[move_choice]
            else:
                self.turn_around()

        self.time += 1
        if self.time > full_str_pheromone_time:
            self.pheromone_strenght = self.pheromone_strenght*beta
        else:
            self.pheromone_strenght = 1
        return food_counter


# %%
L1, L2 = 70, 70
world = np.zeros((L1, L2, 3), dtype=np.float16)
#food - 1, nest - 2, 3 - ant, 4 - blocked
nest_coord = [L1//2, L2//2]
world[L1//2, L2//2, 0] = 2
world[1:7, 6:9, 0] = 1
world[60:68, 52:60, 0] = 1
world[40:60, 36:50, 0] = 4
world[10:12, 5:35, 0] = 4
world[10:12, 40:60, 0] = 4
plt.imshow(world[:, :, 0])
plt.show()
ants_num = 50
ants_list = []
T_max = 5000


def show_world(world, ants_list, t=0, filenames=[], save=False):
    plt.imshow(world[:, :, 0])
    plt.imshow(world[:, :, 1], cmap='Greens', alpha=0.3)
    plt.imshow(world[:, :, 2], cmap='Reds', alpha=0.3)
    poslist = [ant.pos for ant in ants_list]
    poslist = np.array(poslist)
    plt.scatter(poslist[:, 1], poslist[:, 0], s=0.5, c='black')
    if save:
        plt.savefig(f"./imgs/ants_{t}.png")
        filenames.append(f"./imgs/ants_{t}.png")
        plt.close()
    else:
        plt.show()


# %%
dir_choices = np.random.choice(len(angle_list), ants_num)
food_counter = 0
filenames = []
foodlist = []
alpha = 0.995  # evaporation constant
for num in range(ants_num):

    a = ant(angle_list[dir_choices[num]], nest_coord, (L1, L2))
    ants_list.append(a)
    for ente in ants_list:
        food_counter = ente.make_move(world, food_counter)
    world[:, :, 1] *= alpha
    world[:, :, 2] *= alpha
    foodlist.append(food_counter)
    show_world(world, ants_list)
    # show_world(world, ants_list, num, filenames, save = True)
# %%
for t in range(T_max):
    for ente in ants_list:
        food_counter = ente.make_move(world, food_counter)
    world[:, :, 1] *= alpha
    world[:, :, 2] *= alpha
    foodlist.append(food_counter)
    # show_world(world, ants_list)
    if (num+t) % 10 == 0:
        # if False:
        show_world(world, ants_list, ants_num+t, filenames, save=True)
# %%
plt.plot(foodlist)
plt.show()

with imageio.get_writer('./t1.gif', mode='I', duration=40) as writer:
    for frame in filenames:
        image = imageio.imread(frame)
        writer.append_data(image)
writer.close()
# %%

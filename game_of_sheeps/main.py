"""
Python game of life
"""

import numpy as np
import matplotlib.pyplot as plt


class Map:
    def __init__(self, width=30, height=50, peak_centre=None, nsheep=0, nwolves=0):

        # Map
        self.width = width
        self.height = height
        shape = (height, width)

        # parameters
        self.water_multiplier = 3  # 3 * width * height
        self.lakewater = 5
        self.grass_food = 5
        self.sheep_food = 10 * self.grass_food
        self.wolf_food = 3 * self.sheep_food
        self.new_sheep_chance = 0.1
        self.new_wolf_chance = 0.01
        self.cloud_water = 0
        self.cloud_max = (self.water_multiplier * width * height) // 3
        self.debug = 0  # 0, 1(Map), 2(animal), 3(animal action)
        self.new_sheep = 0
        self.new_wolf = 0
        self.sheep_id = 1
        self.wolf_id = 1

        # Type map
        # 0=Empty, 1=Sheep, 2=Wolf
        self.typemap = np.zeros((height, width), dtype=int)

        # Animals
        self.sheep_list = []
        for n in range(nsheep):
            i, j = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
            Sheep(self, i, j)
        self.wolf_list = []
        for n in range(nwolves):
            i, j = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
            Wolf(self, i, j)

        # History
        self.time_stamp = 0
        self.water_drunk = 0
        self.grass_eaten = 0
        self.sheep_died = 0
        self.sheep_eaten = 0
        self.sheep_reproduced = 0
        self.wolves_died = 0
        self.wolves_reproduced = 0

        if peak_centre is None:
            peak_centre = (np.random.randint(height + 5) - 5, np.random.randint(width + 5) - 5)

        self.xx, self.yy = np.meshgrid(range(width), range(height))
        dxx = self.xx - peak_centre[1]
        dyy = self.yy - peak_centre[0]
        dist2peak = np.sqrt(dxx ** 2 + dyy ** 2)
        ht = 10
        self.heightmap = ht ** 4 / (dist2peak + ht ** 2) ** 2  # heighmap has max 1.0

        # Determine index of each tile's neighbor at lower height
        neighbor_dir = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1]
        ])
        roll_height = np.array([
            self.heightmap,
            np.roll(self.heightmap, 1, 0),  # shift, axis
            np.roll(self.heightmap, 1, (0, 1)),
            np.roll(self.heightmap, 1, 1),
            np.roll(np.roll(self.heightmap, -1, 0), 1, 1),  # diagonal
            np.roll(self.heightmap, -1, 0),
            np.roll(self.heightmap, -1, (0, 1)),
            np.roll(self.heightmap, -1, 1),
            np.roll(np.roll(self.heightmap, 1, 0), -1, 1),
        ])
        min_height_idx = np.argmax(roll_height, axis=0)
        self.downhillmap = np.zeros((height, width, 2), dtype=int)
        for h in range(height):
            for w in range(width):
                self.downhillmap[h, w, 0] = h + neighbor_dir[min_height_idx[h, w], 0]
                self.downhillmap[h, w, 1] = w + neighbor_dir[min_height_idx[h, w], 1]
        self.downhillmap[self.downhillmap[:, :, 0] >= height, 0] = height - 1
        self.downhillmap[self.downhillmap[:, :, 1] >= width, 1] = width - 1
        self.downhillmap[self.downhillmap[:, :, 0] < 0, 0] = 0
        self.downhillmap[self.downhillmap[:, :, 1] < 0, 1] = 0

        idx = np.argsort(self.heightmap, axis=None)[::-1]  # sort decending as flattened array
        self.high2low = np.array(np.unravel_index(idx, self.heightmap.shape)).T

        self.watermap = 3 * np.ones(shape)
        self.foodmap = 1.0 * (np.random.rand(height, width) > 0.6)
        # remove food at edges
        self.foodmap[0, :] = 0
        self.foodmap[-1, :] = 0
        self.foodmap[:, 0] = 0
        self.foodmap[:, -1] = 0
        self.grassmap = np.zeros(shape)
        self.image = None

    def water_move(self):
        new_watermap = 1.0 * self.watermap  # avoid multiple movements of water
        for i, j in self.high2low:
            water = self.watermap[i, j]
            ni, nj = self.downhillmap[i, j, :]
            # print(i, j, water, ni, nj)
            downhill_water = self.watermap[ni, nj]
            if downhill_water <= water and water > 0:
                new_watermap[i, j] -= 1
                new_watermap[ni, nj] += 1
        self.watermap = new_watermap

    def evaporate(self):
        """Water in lake evaporates into cloud"""
        lakewater = (self.watermap >= self.lakewater) * (np.random.rand(self.height, self.width) > 0.9)

        self.cloud_water += np.sum(lakewater)
        self.watermap[lakewater] -= 1

    def rain(self):
        """Rain chance"""
        rain_chance = self.cloud_water / (1.0 * self.cloud_max)
        if rain_chance > np.random.rand():
            # Rain
            if self.debug:
                print('Raining! %d water' % self.cloud_water)
            rain_i = np.random.randint(self.height, size=self.cloud_water)
            rain_j = np.random.randint(self.width, size=self.cloud_water)
            for i, j in zip(rain_i, rain_j):
                self.watermap[i, j] += 1
            self.cloud_water = 0

    def grow_grass(self):
        """water + food = grass"""
        new_grass = (self.watermap > 0) * (self.watermap < self.lakewater) * (self.grassmap < self.grass_food) * (
                    self.foodmap > 0)
        self.grassmap[new_grass] += 1
        if self.debug:
            print('New Grass: %d, max grass: %d' % (np.sum(new_grass), np.max(self.grassmap)))
        # self.watermap[new_grass] -= 1
        # self.foodmap[new_grass]

    def create_sheep(self):
        """create sheep from excess food"""
        excess_water = np.sum(self.watermap[self.watermap > self.lakewater] - self.lakewater)
        excess_food = np.sum(self.foodmap[self.foodmap > 1]-1) + np.sum(self.foodmap[self.watermap >= self.lakewater])
        if excess_water > 0 and excess_food > 0 and np.random.rand() < self.new_sheep_chance:
            i, j = (np.random.randint(2, self.height - 2), np.random.randint(2, self.width - 2))
            sh = Sheep(self, i, j, food=excess_food, water=excess_water)
            self.watermap[self.watermap > self.lakewater] = self.lakewater
            self.foodmap[self.foodmap > 1] = 1
            self.foodmap[self.watermap >= self.lakewater] = 0
            print('New Sheep %s' % sh)
            self.new_sheep += 1
            if self.debug:
                print('New Sheep %s' % sh)

    def create_wolf(self):
        """create sheep from excess food"""
        excess_water = np.sum(self.watermap[self.watermap > self.lakewater] - self.lakewater)
        excess_food = np.sum(self.foodmap[self.foodmap > 1] - 1) + np.sum(self.foodmap[self.watermap >= self.lakewater])
        if excess_water > 0 and excess_food > 0 and np.random.rand() < self.new_wolf_chance:
            i, j = (np.random.randint(2, self.height - 2), np.random.randint(2, self.width - 2))
            sh = Wolf(self, i, j, food=excess_food, water=excess_water)
            self.watermap[self.watermap > self.lakewater] = self.lakewater
            self.foodmap[self.foodmap > 1] = 1
            self.foodmap[self.watermap >= self.lakewater] = 0
            print('New Wolf %s' % sh)
            self.new_wolf += 1
            if self.debug:
                print('New Wolf %s' % sh)

    def time(self):
        """Move time by 1"""
        self.time_stamp += 1
        self.grow_grass()
        self.water_move()
        self.evaporate()
        self.rain()
        for sheep in self.sheep_list:
            sheep.act()
        for wolf in self.wolf_list:
            wolf.act()
        self.create_sheep()
        self.create_wolf()

        tot_water = np.sum(self.watermap)
        tot_food = np.sum(self.foodmap)
        tot_lake = np.sum(self.watermap >= self.lakewater)
        tot_grass = np.sum(self.grassmap >= self.grass_food)
        tot_sheep = len(self.sheep_list)
        tot_wolf = len(self.wolf_list)
        fmt = "%4d water: %5.0f, cloud: %3.0f, lake: %3.0f, food: %4.0f, grass: %3.0f, sheep: %1.0f, wolves: %1.0f"
        if self.debug:
            print(fmt % (
            self.time_stamp, tot_water, self.cloud_water, tot_lake, tot_food, tot_grass, tot_sheep, tot_wolf))

    def position_allowed(self, i, j):
        """Is given position allowed to move"""
        if i < 1 or i >= (self.height - 1):
            return False
        if j < 1 or j >= (self.width - 1):
            return False
        if self.watermap[i, j] >= self.lakewater:
            return False
        if self.typemap[i, j] > 0:
            # no overlapping animals
            return False
        return True

    def distancemap(self, i, j):
        """returns map of distances from i,j"""
        return np.sqrt((self.xx - j) ** 2 + (self.yy - i) ** 2)

    def nearest_lake(self, i, j):
        """return position next to i,j in direction of water"""
        lakes = self.watermap >= self.lakewater
        dist = self.distancemap(i, j)
        # nearest lake
        ni, nj = np.unravel_index(np.argmax(lakes / (dist + 0.5)), lakes.shape)
        return ni, nj

    def nearest_grass(self, i, j):
        """return position next to i,j in direction of grass"""
        grass = self.grassmap >= self.grass_food
        dist = self.distancemap(i, j)
        # nearest grass
        ni, nj = np.unravel_index(np.argmax(grass / (dist + 0.5)), grass.shape)
        return ni, nj

    def nearest_sheep(self, i, j):
        """return position next to i,j in direction of sheep"""
        dist = self.distancemap(i, j)
        # sheepmap = np.zeros(dist.shape)
        # for sheep in self.sheep_list:
        #    sheepmap[sheep.position] = 1
        sheepmap = self.typemap == 1
        # nearest grass
        ni, nj = np.unravel_index(np.argmax(sheepmap / (dist + 0.5)), sheepmap.shape)
        return ni, nj

    def nearest_wolf(self, i, j):
        """return position next to i,j in direction of sheep"""
        dist = self.distancemap(i, j)
        # wolfmap = np.zeros(dist.shape)
        # for wolf in self.wolf_list:
        #    wolfmap[wolf.position] = 1
        wolfmap = self.typemap == 2
        # nearest grass
        ni, nj = np.unravel_index(np.argmax(wolfmap / (dist + 0.5)), wolfmap.shape)
        return ni, nj

    def create_image(self):
        # heightmap has max 1, min > 0
        image = plt.cm.copper(self.heightmap)  # (height, width, 4)
        # colour grass
        grass = self.grassmap >= self.grass_food
        image[grass, :] = (0, 1, 0, 1)
        # colour water
        image[self.watermap >= self.lakewater, :] = (0, 0, 1, 1)  # RGBA
        # Colour sheep
        for sheep in self.sheep_list:
            i, j = sheep.position
            image[i, j, :] = (1, 1, 1, 1)
        for wolf in self.wolf_list:
            i, j = wolf.position
            image[i, j, :] = (0, 0, 0, 1)
        return image

    def create_plot(self):
        image = self.create_image()

        plt.figure(figsize=[10, 8], dpi=60)
        self.image = plt.imshow(image)

    def update_plot(self):
        if self.image is not None:
            image = self.create_image()
            self.image.set_data(image)
            plt.draw()

    def __str__(self):
        map_water = np.sum(self.watermap)
        map_food = np.sum(self.foodmap)
        tot_lake = np.sum(self.watermap >= self.lakewater)
        tot_grass = np.sum(self.grassmap >= self.grass_food)
        tot_sheep = len(self.sheep_list)
        tot_wolf = len(self.wolf_list)
        wolf_water = np.sum([w.water for w in self.wolf_list])
        sheep_water = np.sum([s.water for s in self.sheep_list])
        wolf_food = np.sum([w.food for w in self.wolf_list])
        sheep_food = np.sum([s.food for s in self.sheep_list])
        tot_water = map_water + self.cloud_water + sheep_water + wolf_water
        tot_food = map_food + sheep_food + wolf_food

        excess_water = np.sum(self.watermap[self.watermap > self.lakewater])
        excess_food = np.sum(self.foodmap[self.foodmap > 1]) + np.sum(self.foodmap[self.watermap >= self.lakewater])

        out = 'Map(%d, %d) time=%d\n' % (self.height, self.width, self.time_stamp)
        out += ' Water total: %5.0f, map: %5.0f, cloud: %3.0f, ' % (tot_water, map_water, self.cloud_water)
        out += 'lakes: %3.0f, animal_water: %3.0f\n' % (tot_lake, sheep_water + wolf_water)
        out += ' Food total: %5.0f, map: %5.0f, grass: %3.0f, ' % (tot_food, map_food, tot_grass)
        out += 'animal_food: %3.0f\n' % (sheep_food + wolf_food)
        out += ' Excess water: %3d, food: %3d\n' % (excess_water, excess_food)
        out += ' Animals: %3d sheep, %3d wolves\n' % (tot_sheep, tot_wolf)
        out += '  New animals: %3d sheep, %3d wolves\n' % (self.new_sheep, self.new_wolf)
        out += ' Actions:\n'
        out += '  water drunk: %1.0f\n' % self.water_drunk
        out += '  grass eaten: %1.0f\n' % self.grass_eaten
        out += '  sheep eaten: %1.0f\n' % self.sheep_eaten
        out += '  sheep repro: %1.0f\n' % self.sheep_reproduced
        out += '  sheep died : %1.0f\n' % self.sheep_died
        out += '  wolves repr: %1.0f\n' % self.wolves_reproduced
        out += '  wolves died: %1.0f\n' % self.wolves_died
        return out


#####################################ANIMALS##################################


class Wolf:
    """Living thing"""

    def __init__(self, world, i, j, food=10, water=30, max_food=60, max_water=100):
        self.map = world
        self.position = (i, j)  # height, width
        self.food = food
        self.water = water
        self.max_food = max_food
        self.max_water = max_water
        self.poo_chance = 0.3
        self.wee_chance = 0.1
        self.hungry = 0.35  # hunts when food < hungry*maxfood
        self.is_dead = False
        self.map.wolf_list += [self]
        self.looking_for_water = 0
        self.looking_for_food = 0
        self.id = 1*self.map.wolf_id
        self.map.wolf_id += 1
        self.type = 2  # number in typemap

    def __str__(self):
        if self.looking_for_food:
            state = 'Hunting'
        elif self.looking_for_water:
            state = 'Thirsty'
        else:
            state = 'Dawdling'
        s = 'Wolf(%3d: %2d,%2d) Water %3.0f, Food: %3.0f, State: %s'
        return s % (self.id, *self.position, self.water, self.food, state)

    def nearest_move(self, i, j, move_away=False):
        """returns movement by one in direction of (i,j)"""
        if move_away:
            di, dj = self.position[0] - i, self.position[1] - j
        else:
            di, dj = i - self.position[0], j - self.position[1]
        move_i = np.round(di / np.sqrt(di ** 2 + dj ** 2)).astype(int)
        move_j = np.round(dj / np.sqrt(di ** 2 + dj ** 2)).astype(int)
        return move_i, move_j

    def move(self):
        if self.water < self.food:
            # move to water
            ni, nj = self.map.nearest_lake(*self.position)
            i, j = self.nearest_move(ni, nj)
            self.looking_for_water = 1
            self.looking_for_food = 0
        elif self.food < self.max_food * self.hungry and len(self.map.sheep_list) > 0:
            # move to sheep
            ni, nj = self.map.nearest_sheep(*self.position)
            i, j = self.nearest_move(ni, nj)
            self.looking_for_food = 1
            self.looking_for_water = 0
            if self.map.debug> 2:
                print('wolf hunting %d,%d, sheep: %d,%d, move: %d,%d' % (*self.position, ni, nj, i, j))
        else:
            # Random move
            i, j = np.random.randint(-1, 2, size=2)  # -1,0,1
            self.looking_for_food = 0
            self.looking_for_water = 0

        new_i = self.position[0] + i
        new_j = self.position[1] + j
        if self.map.position_allowed(new_i, new_j):
            self.map.typemap[self.position] = 0
            self.position = (new_i, new_j)
            self.map.typemap[self.position] = self.type
        elif self.map.watermap[self.position[0], self.position[1]] >= self.map.lakewater:
            # In lake
            if self.map.debug > 1:
                print('  Wolf drowned, moving to middle')
            self.map.typemap[self.position] = 0
            self.position = (self.map.height // 2, self.map.width // 2)
            self.map.typemap[self.position] = self.type

    def drink(self):
        if self.water >= self.max_water:
            return
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                pos = (self.position[0] + i, self.position[1] + j)
                if self.map.watermap[pos] >= self.map.lakewater:
                    self.water += 2
                    self.map.watermap[pos] -= 2
                    self.map.water_drunk += 2
                    if self.map.debug > 2:
                        print('    wolfgulp: water: %d, mapwater: %d' % (self.water, self.map.watermap[self.position]))
                    return

    def eat_sheep(self):
        i, j = self.position
        for sheep in self.map.sheep_list:
            si, sj = sheep.position
            if abs(si - i) <= 1 and abs(sj - j) <= 1:
                self.food += sheep.food
                sheep.food = 0
                sheep.die()
                self.map.sheep_eaten += 1
                if self.map.debug > 2:
                    print('    chomp on sheep: food: %d' % self.food)
                return

    def poo(self):
        if self.poo_chance > np.random.rand():
            self.food -= 1
            self.map.foodmap[self.position] += 1
            if self.map.debug > 2:
                print('    WolfPlop: food: %d, mapfood: %d, mapgrass: %d' % (
                self.food, self.map.foodmap[self.position], self.map.grassmap[self.position]))

    def wee(self):
        if self.wee_chance > np.random.rand():
            self.water -= 1
            self.map.watermap[self.position] += 1
            if self.map.debug > 2:
                print('    hiss: water: %d, mapwater: %d' % (self.water, self.map.watermap[self.position]))

    def reproduce(self):
        if self.food >= self.max_food:
            i, j = self.position
            for ni, nj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                if self.map.position_allowed(i + ni, j + nj):
                    if self.map.debug > 1:
                        print('    Cub time!')
                    new_food = self.food // 2
                    new_water = self.water // 2
                    lamb = Wolf(self.map, i + ni, j + nj, food=new_food, water=new_water, max_food=self.max_food)
                    self.food -= new_food
                    self.water -= new_water
                    self.map.wolves_reproduced += 1
                    return

    def die(self):
        if self.food <= 0 or self.water <= 0:
            if self.map.debug > 1:
                print('Wolf Dead')
            self.map.foodmap[self.position] += self.food
            self.map.watermap[self.position] += self.water
            self.is_dead = True
            self.map.wolf_list.remove(self)
            self.map.typemap[self.position] = 0
            self.map.wolves_died += 1

    def act(self):
        if self.map.debug > 2:
            i, j = self.position
            print('  iWolf(%2d,%2d) food: %2.0f water: %2.0f' % (i, j, self.food, self.water))
        self.move()
        self.drink()
        self.eat_sheep()
        self.wee()
        self.poo()
        self.reproduce()
        self.die()
        if self.map.debug > 1:
            i, j = self.position
            print('  Wolf(%2d,%2d) food: %2.0f water: %2.0f' % (i, j, self.food, self.water))


class Sheep:
    """Living thing"""

    def __init__(self, world, i, j, food=10, water=30, max_food=20, max_water=100):
        self.map = world
        self.position = (i, j)  # height, width
        self.type = 1  # number in typemap
        self.food = food
        self.water = water
        self.max_water = max_water
        self.max_food = max_food
        self.poo_chance = 0.05
        self.wee_chance = 0.2
        self.hungry = 0.4  # grazes when food < hungry*maxfood
        self.sight_distance = 3  # runs from wolf when wolf within sight_distance
        self.is_dead = False
        self.map.sheep_list += [self]
        self.looking_for_water = 0
        self.looking_for_food = 0
        self.running_from_wolves = 0
        self.id = 1 * self.map.sheep_id
        self.map.sheep_id += 1

    def __str__(self):
        if self.looking_for_food:
            state = 'Grazing'
        elif self.looking_for_water:
            state = 'Thirsty'
        else:
            state = 'Dawdling'
        s = 'Sheep(%3d: %2d,%2d) Water %3.0f, Food: %3.0f, State: %s'
        return s % (self.id, *self.position, self.water, self.food, state)

    def nearest_move(self, i, j, move_away=False):
        """returns movement by one in direction of (i,j)"""
        if move_away:
            di, dj = self.position[0] - i, self.position[1] - j
        else:
            di, dj = i - self.position[0], j - self.position[1]
        move_i = np.round(di / np.sqrt(di ** 2 + dj ** 2)).astype(int)
        move_j = np.round(dj / np.sqrt(di ** 2 + dj ** 2)).astype(int)
        return move_i, move_j

    def move(self):
        ni, nj = self.map.nearest_wolf(*self.position)
        wolfdist = np.sqrt((ni - self.position[0]) ** 2 + (nj - self.position[1]) ** 2)
        if len(self.map.wolf_list) > 0 and wolfdist < self.sight_distance:
            # running from wolves
            i, j = self.nearest_move(ni, nj, move_away=True)
            # i=-i
            # j=-j

            if self.map.debug> 2:
                print('running from wolves %d,%d (%3.2f), wolf: %d,%d, move: %d,%d' % (
                *self.position, wolfdist, ni, nj, i, j))
            self.running_from_wolves = 1
            self.looking_for_water = 0
            self.looking_for_food = 0
        elif self.water < self.food:
            # move to water
            ni, nj = self.map.nearest_lake(*self.position)
            i, j = self.nearest_move(ni, nj)
            self.looking_for_water = 1
            self.looking_for_food = 0
            self.running_from_wolves = 0
            if self.map.debug> 2:
                print('Sheep drinking', i, j)
        elif self.food < self.max_food * self.hungry:
            # move to sheep
            ni, nj = self.map.nearest_grass(*self.position)
            i, j = self.nearest_move(ni, nj)
            self.looking_for_food = 1
            self.looking_for_water = 0
            self.running_from_wolves = 0
            if self.map.debug> 2:
                print('Sheep Grazing', i, j)
        else:
            # Random move
            i, j = np.random.randint(-1, 2, size=2)  # -1,0,1
            self.looking_for_food = 0
            self.looking_for_water = 0
            self.running_from_wolves = 0
            if self.map.debug> 2:
                print('Sheep dawdling', i, j)

        new_i = self.position[0] + i
        new_j = self.position[1] + j
        if self.map.position_allowed(new_i, new_j):
            self.map.typemap[self.position] = 0
            self.position = (new_i, new_j)
            self.map.typemap[self.position] = self.type
        elif self.map.watermap[self.position[0], self.position[1]] >= self.map.lakewater:
            # In lake
            if self.map.debug > 1:
                print('  Sheep drowned, moving to middle')
            self.map.typemap[self.position] = 0
            self.position = (self.map.height // 2, self.map.width // 2)
            self.map.typemap[self.position] = self.type

    def drink(self):
        if self.water >= self.max_water:
            return
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                pos = (self.position[0] + i, self.position[1] + j)
                if self.map.watermap[pos] >= self.map.lakewater:
                    self.water += 1
                    self.map.watermap[pos] -= 1
                    self.map.water_drunk += 1
                    if self.map.debug > 2:
                        print('    gulp: water: %d, mapwater: %d' % (self.water, self.map.watermap[self.position]))
                    return

    def eat_grass(self):
        if self.map.grassmap[self.position] >= self.map.grass_food:
            if self.map.debug > 2:
                print('    Nom nom: food: %d, mapfood: %d, mapgrass: %d' % (
                self.food, self.map.foodmap[self.position], self.map.grassmap[self.position]))
            self.food += 1
            self.map.foodmap[self.position] -= 1
            self.map.grassmap[self.position] -= self.map.grass_food
            self.map.grass_eaten += 1

    def poo(self):
        if self.poo_chance > np.random.rand():
            self.food -= 1
            self.map.foodmap[self.position] += 1
            if self.map.debug > 2:
                print('    Plop: food: %d, mapfood: %d, mapgrass: %d' % (
                self.food, self.map.foodmap[self.position], self.map.grassmap[self.position]))

    def wee(self):
        if self.wee_chance > np.random.rand():
            self.water -= 1
            self.map.watermap[self.position] += 1
            if self.map.debug > 2:
                print('    hiss: water: %d, mapwater: %d' % (self.water, self.map.watermap[self.position]))

    def reproduce(self):
        if self.food >= self.max_food:
            i, j = self.position
            for ni, nj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                if self.map.position_allowed(i + ni, j + nj):
                    if self.map.debug > 1:
                        print('    Lamb time!')
                    new_food = self.food // 2
                    new_water = self.water // 2
                    lamb = Sheep(self.map, i + ni, j + nj, food=new_food, water=new_water, max_food=self.max_food)
                    self.food -= new_food
                    self.water -= new_water
                    self.map.sheep_reproduced += 1
                    return

    def die(self):
        if self.food <= 0 or self.water <= 0:
            if self.map.debug > 1:
                print('Sheep Dead')
            self.map.foodmap[self.position] += self.food
            self.map.watermap[self.position] += self.water
            self.is_dead = True
            self.map.sheep_list.remove(self)
            self.map.typemap[self.position] = 0
            self.map.sheep_died += 1

    def act(self):
        if self.map.debug > 2:
            i, j = self.position
            print('  iSheep(%2d,%2d) food: %2.0f water: %2.0f' % (i, j, self.food, self.water))
        self.move()
        self.drink()
        self.eat_grass()
        self.wee()
        self.poo()
        self.reproduce()
        self.die()
        if self.map.debug > 1:
            i, j = self.position
            print('  Sheep(%2d,%2d) food: %2.0f water: %2.0f' % (i, j, self.food, self.water))


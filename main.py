"""
Game of Sheeps
"""

import numpy as np
import matplotlib.pyplot as plt
from game_of_sheeps import Map, SheepsGui

sg = SheepsGui(nsheep=0, nwolves=0)


"""
m = Map(nsheep=30, nwolves=1)
print(m)

im_time = 50
iterations = 10
sheep_water = []
wolf_water = []
sheep_food = []
wolf_food = []
nsheep = []
nwolves = []
sheep_grazing = []
wolves_hunting = []
for nn in range(iterations):
    for n in range(im_time):
        m.time()
        nsheep += [len(m.sheep_list)]
        nwolves += [len(m.wolf_list)]
        sheep_water += [np.sum([s.water for s in m.sheep_list])]
        wolf_water += [np.sum([w.water for w in m.wolf_list])]
        sheep_food += [np.sum([s.food for s in m.sheep_list])]
        wolf_food += [np.sum([w.food for w in m.wolf_list])]
        sheep_grazing += [np.sum([s.looking_for_food for s in m.sheep_list])]
        wolves_hunting += [np.sum([w.looking_for_food for w in m.wolf_list])]
    print(m)
    m.create_plot()
    plt.show()

plt.figure(figsize=[12, 8], dpi=60)
plt.subplot(2, 1, 1)
plt.plot(nsheep, 'k-o', lw=2, ms=6, label='Sheep')
plt.plot(nwolves, 'k-x', lw=2, ms=8, label='Wolves')
plt.plot(sheep_grazing, 'go', ms=6, label='Sheep grazing')
plt.plot(wolves_hunting, 'rx', ms=8, label='Wolves hunting')
plt.legend(loc=0, frameon=False)

plt.subplot(2, 1, 2)
plt.plot(sheep_water, 'b-o', lw=2, ms=6, label='Sheep water')
plt.plot(wolf_water, 'b-x', lw=2, ms=8, label='Wolf water')
plt.plot(sheep_food, 'g-o', lw=2, ms=6, label='Sheep food')
plt.plot(wolf_food, 'g-x', lw=2, ms=8, label='Wolf food')
plt.xlabel('time')
plt.legend(loc=0, frameon=False)
plt.show()
"""






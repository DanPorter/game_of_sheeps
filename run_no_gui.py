"""
Run simulation with no GUI, just text output
"""

from game_of_sheeps import Map

map = Map(nsheep=0, nwolves=0)
print(map)

for n in range(300):
    map.time()
    print(map)
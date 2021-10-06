from Base.bp2DBox import Box
from BoxPlacementEnvironment import BoxPlacementEnvironment
from Base.bpReadWrite import ReadWrite

state = ReadWrite.read_state("test_instances/test_1_input")

env = BoxPlacementEnvironment(state)

print(env.reset().shape)

print("Initial state:")
print(env._next_observation())
print("--------------")

print("Try to place inexistent box (should be unmodified)")
env.step([0, 1, 2, 0, 0])
print(env._next_observation())
print("--------------")

print("Try to place box that intersects walls (should be unmodified)")
env.step([0, 1, 1, 9, 9])
print(env._next_observation())
print("--------------")

print("Place box correctly")
env.step([0, 1, 1, 0, 0])
print(env._next_observation())
print("--------------")

print("Try to place box over box")
env.step([0, 2, 3, 1, 1])
print(env._next_observation())
print("--------------")

print("New bin")
env.step([1, 0, 0, 0, 0])
print(env._next_observation())
print("--------------")

print("Reset")
env.reset()
print(env._next_observation())
print("--------------")

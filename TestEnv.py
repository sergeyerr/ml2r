from Base.bp2DBox import Box
from BoxPlacementEnvironment import BoxPlacementEnvironment
from Base.bpReadWrite import ReadWrite

state = ReadWrite.read_state("test_instances/test_1_input")

env = BoxPlacementEnvironment(state)

print("Initial state:")
print(env.bin_occupation)
print(env.box_counts)
print("--------------")

print("Try to place inexistent box (should be unmodified)")
env.step(1)
print(env.bin_occupation)
print(env.box_counts)
print("--------------")

print("Place box correctly")
env.step(1 + 2 * 10 + 3)
print(env.bin_occupation)
print(env.box_counts)
print("--------------")

print("Try to place box over box")
env.step(1 + 1 * 10 + 1)
print(env.bin_occupation)
print(env.box_counts)
print("--------------")

print("New bin")
env.step(0)
print(env.bin_occupation)
print(env.box_counts)
print("--------------")

print("Reset")
env.reset()
print(env.bin_occupation)
print(env.box_counts)
print("--------------")

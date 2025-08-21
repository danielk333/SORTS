#!/usr/bin/env python

"""
Profiling memory leaks
=========================

"""
import matplotlib.pyplot as plt

# TODO: revisit what we want to do with memory profiling
#   notes from danielk:
#     The memory profiling im still not entiarly satisfied with - if we refactor the profiler to
#     JUST do memory profiling (and use yappi for the runtime profiling) then it might be useful
#     for debugging long running dynamic simulations - so lets remove this and put that one a todo

# As the profiler data is also stored in Python tracked memory
# a diff of "nothing" will still result in more allocation of memory
# including initialization for structures and the like
p.snapshot("nothing")
p.memory_diff("nothing")

# this allocates memory
p.snapshot("one list")
lst = list(range(2000))
p.memory_diff("one list")

# and if we take care to delete the variable
# only profiling allocations are left over
del lst
p.memory_diff("one list", save="one list - and clear")

# this iteration changes allocation each iteration and does not clean up
lsts = []
for i in range(1000):
    p.snapshot("dynamic string creation")
    lsts.append("test" * i)
    p.memory_diff("dynamic string creation")


print(p)

# so it might be a good idea to plot the trend
fig, ax = plt.subplots(1, 1)
ax.plot(p.memory_stats["dynamic string creation"])
ax.set_title("Dynamic string creation in loop")
ax.set_xlabel("Iteration")
ax.set_ylabel("Allocation [kB]")
plt.show()

#!/usr/bin/env python

'''
=======================
Profiling save and load
=======================

Showcases the saving/loading features implemented in the ``sorts.profiling`` module.

The example evaluates and saves the performances of a simple loop, and then generates a new Profiler instance based 
on the previously saved results.
'''
import pathlib
from sorts.common.profiling import Profiler

# performances evaluation loop
p = Profiler()
p.start('program')

p.start('list init')
lst = list(range(200))
p.stop('list init')

for i in range(1000):
    p.start('list reversal')
    lst = lst[::-1]
    p.stop('list reversal')

p.stop('program')

print(p)

# save profiling results
pth = pathlib.Path(__file__).parent / 'data' / 'profiler_data.txt'
print(f'Writing profiler data to: {pth}')
p.to_txt(pth)

# load profiling results
new_profiler = Profiler.from_txt(pth)

print(new_profiler)
#!/usr/bin/env python

'''
Profiling save and load
================================

'''
import pathlib
from sorts.profiling import Profiler

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


pth = pathlib.Path(__file__).parent / 'data' / 'profiler_data.txt'
print(f'Writing profiler data to: {pth}')
p.to_txt(pth)

new_profiler = Profiler.from_txt(pth)

print(new_profiler)
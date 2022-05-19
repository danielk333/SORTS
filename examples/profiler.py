#!/usr/bin/env python

'''
Profiling
==========

'''
from sorts.common.profiling import Profiler

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
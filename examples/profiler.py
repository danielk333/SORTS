#!/usr/bin/env python

'''
=========
Profiling
=========

Showcases the ``sorts.profiling`` module defined within ``sorts`` by
adding multiple profining entries and plotting the resutls
'''
from sorts.common.profiling import Profiler

# initializes the profiler
p = Profiler()
p.start('program')

# create a list of 200 entries
p.start('list init')
lst = list(range(200))
p.stop('list init')

# reverse the list
for i in range(1000):
    p.start('list reversal')
    lst = lst[::-1]
    p.stop('list reversal')

p.stop('program')

# print performances
print(p)
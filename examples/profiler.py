#!/usr/bin/env python

"""
Profiling
==========

"""

# ASK: this is an example file for using/testing the profiler, shall we remove the whole thing?
# RES: Yes, instead we should show an example of profiling with yappi

from sorts.profiling import Profiler

p = Profiler()
p.start("program")

p.start("list init")
lst = list(range(200))
p.stop("list init")

for i in range(1000):
    p.start("list reversal")
    lst = lst[::-1]
    p.stop("list reversal")

p.stop("program")

print(p)

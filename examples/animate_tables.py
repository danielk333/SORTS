#!/usr/bin/env python

'''
Animate tables
===============

'''

import numpy as np

import sorts

data = np.random.randn(100)
table = [[j, x] for j, x in enumerate(data)]

for i in range(0,95):
    sorts.io.step_flush_table(
        table[i:(i+10)], 
        header = ['index', 'random number'], 
        table_size = 10, 
        first_step = i==0, 
        step_time = 0.1,
    )

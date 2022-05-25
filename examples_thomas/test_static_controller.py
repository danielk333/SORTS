#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:54:50 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

from sorts.radar.system import instances
from sorts.radar import scheduler, controllers
from sorts import equidistant_sampling

from sorts.common import profiling
from sorts import plotting

import pyorb

# Computation / test setup
end_t = 24*3600
nbplots = 3

end_t = 10


# RADAR definition
eiscat3d = instances.eiscat3d
    
# Profiler
p = profiling.Profiler()
logger = profiling.get_logger('scanning')

p.start("Total")
static_controller = controllers.Static(profiler=p, logger=logger)

t_slice = 0.1
t = np.arange(0, end_t, t_slice)
controls = static_controller.generate_controls(t, eiscat3d, t_slice=t_slice, max_points=100)

# plot the generated controls
p.start("examples_main:retreiving_control_values")

plt_ids = np.linspace(0, int(end_t/t_slice), nbplots, dtype=int)

print(plt_ids)

for i in range(len(controls["beam_orientation"])):
    ctrl = next(controls["beam_orientation"][i])
    print(f"control sub-array {i} size : {(ctrl['tx'].itemsize*np.size(ctrl['tx']) + ctrl['rx'].itemsize*np.size(ctrl['rx']))/1e6} Mb")
    
    if i in plt_ids:
        plotting.plot_beam_directions(ctrl, eiscat3d, logger=logger, profiler=p, zoom_level=0.95)

    del ctrl
    
p.stop("examples_main:retreiving_control_values")

p.stop("Total")
print(p)

plt.show()


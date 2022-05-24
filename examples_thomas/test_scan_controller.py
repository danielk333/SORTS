#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:15:39 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

from sorts.radar.scans import Fence
from sorts.radar.system import instances
from sorts.radar import controllers

from sorts.common import profiling
from sorts import plotting

# Computation / test setup
end_t = 60

# Scan type definition
scan = Fence(azimuth=90, min_elevation=30, dwell=0.1, num=50)

# RADAR definition
eiscat3d = instances.eiscat3d
    
# Profiler
p = profiling.Profiler()
logger = profiling.get_logger('scanning')
p.start("Total")

# instanciate the scanning controller 
scanner_ctrl = controllers.Scanner(profiler=p, logger=logger)

# generate the beam orientation controls
t = np.arange(0, end_t, scan.dwell())
controls = scanner_ctrl.generate_controls(t, eiscat3d, scan, priority=-1)

# plot the generated controls
plotting.plot_beam_directions(controls, eiscat3d, logger=logger, profiler=p, zoom_level=1)

p.stop("Total")

plt.show()

print(p)
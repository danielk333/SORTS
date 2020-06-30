#!/usr/bin/env python

'''Example showing how populations can be used

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sorts


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

sorts.plotting.grid_earth(ax)

plt.show()


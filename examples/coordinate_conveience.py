#!/usr/bin/env python

'''
================================
Coordinate convenience functions
================================

Showcases the use of the ``sorts.frames`` module to perform coordinate 
transformations from a direction ``a`` to a direction ``b``.


'''
import numpy as np
import matplotlib.pyplot as plt

import sorts

a = np.array([0,0,1], dtype=np.float64)
b = np.array([1.3,1.1,-4.0], dtype=np.float64)

# coordinate transformation from `a` to `b`
R = sorts.frames.vec_to_vec(a,b)

# create vector directions of the mesh grid
X,Y,Z = np.meshgrid(np.arange(0,5), np.arange(0,5), np.arange(0,5))
vecs = np.empty((3,5**3))
vecs[0,:] = X.flatten()
vecs[1,:] = Y.flatten()
vecs[2,:] = Z.flatten()

# apply transformation to mesh grid points
vecs = R @ vecs
Xp,Yp,Zp = vecs[0,:], vecs[1,:], vecs[2,:]

# apply transformation to a
ap = R @ a

# plot results
fig = plt.figure(figsize=(15,15))
axes = []
axes += [fig.add_subplot(121, projection='3d')]
axes += [fig.add_subplot(122, projection='3d')]

axes[0].plot(X.flatten(), Y.flatten(), Z.flatten(), '.b')
axes[0].plot([0, a[0]], [0, a[1]], [0, a[2]], '-g')
axes[0].plot([0, b[0]], [0, b[1]], [0, b[2]], '-r')
axes[1].plot(Xp, Yp, Zp, '.b')
axes[1].plot([0, a[0]], [0, a[1]], [0, a[2]], '-g')
axes[1].plot([0, b[0]], [0, b[1]], [0, b[2]], '-r')
axes[1].plot([0, ap[0]], [0, ap[1]], [0, ap[2]], '-xm')

plt.show()
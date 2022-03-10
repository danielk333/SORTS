#!/usr/bin/env python

'''General plotting functions

'''


# Third party import
import numpy as np

# Local import
from .. import constants


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    Source: CC BY-SA 3.0 @ https://stackoverflow.com/a/31364297 by karlo

    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True):
    '''Add a 3d spherical grid to the given axis that represent the Earth.

    '''
    lons = np.linspace(-180, 180, num_lon+1) * np.pi/180 
    lons = lons[:-1]
    lats = np.linspace(-90, 90, num_lat) * np.pi/180 

    lonsl = np.linspace(-180, 180, res) * np.pi/180 
    latsl = np.linspace(-90, 90, res) * np.pi/180 

    r_e = constants.R_earth
    for lat in lats:
        x = r_e*np.cos(lonsl)*np.cos(lat)
        y = r_e*np.sin(lonsl)*np.cos(lat)
        z = r_e*np.ones(np.size(lonsl))*np.sin(lat)
        ax.plot(x,y,z,alpha=alpha,linestyle='-', marker='',color=color)

    for lon in lons:
        x = r_e*np.cos(lon)*np.cos(latsl)
        y = r_e*np.sin(lon)*np.cos(latsl)
        z = r_e*np.sin(latsl)
        ax.plot(x,y,z,alpha=alpha,color=color)
    
    if hide_ax:
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()

    return ax

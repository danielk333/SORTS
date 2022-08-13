#!/usr/bin/env python

'''Radar scan plot functions

'''

#Python standard import


#Third party import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from matplotlib import animation

from mpl_toolkits.mplot3d import Axes3D

#Local import
from . import general
from sorts.transformations import frames


# def scan(scan, earth=False, ax=None, max_range=4000e3):
#     '''Plot a full cycle of the scan pattern.
    
#         :param Scan scan: Scan to plot.
#         :param bool earth: Plot the surface of the Earth.
#         :param float max_range: The range of the pointing directions.
#         :param matplotlib.axes ax: The ax to draw the scan on.
#     '''
#     raise NotImplementedError('')

#     min_dwell = scan.min_dwell()
#     cycle = scan.cycle()
    
#     t = np.arange(0.0, cycle, min_dwell)

#     if ax is None:
#         fig = plt.figure(figsize=(15, 15))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.grid(False)
#         ax.view_init(15, 5)
#     else:
#         fig = None

#     if earth:
#         general.grid_earth(ax)
#     plothelp.draw_radar(ax, SC._lat, SC._lon)
    
#     for i in range(len(t)):
#         p0, k0 = SC.antenna_pointing(t[i])
        
#         p1 = p0 + k0*max_range
#         if k0[2] < 0:
#             ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], alpha=0.5, color="red")
#         else:
#             ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], alpha=0.5, color="green")
    
#     if fig is not None:
#         ax.set_xlim(p0[0] - max_range, p0[0] + max_range)
#         ax.set_ylim(p0[1] - max_range, p0[1] + max_range)
#         ax.set_zlim(p0[2] - max_range, p0[2] + max_range)

#     return fig, ax

def plot_scanning_sequence(scan, station=None, earth=False, ax=None, plot_local_normal=False, max_range=4000e3):
    '''Plot a full cycle of the scan pattern.
    
        :param Scan scan: Scan to plot.
        :param bool earth: Plot the surface of the Earth.
        :param float max_range: The range of the pointing directions.
        :param matplotlib.axes ax: The ax to draw the scan on.
    '''
    min_dwell = scan.min_dwell()
    cycle = scan.cycle()
    
    if cycle is None:
        t = np.zeros(1)
    else:
        t = np.arange(0.0, cycle, min_dwell)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
    else:
        fig = None

    if earth:
        general.grid_earth(ax)

    if station is not None: # if station is provided, plot in the ECEF frame
        # plot station
        ax.plot([station.ecef[0]],[station.ecef[1]],[station.ecef[2]],'og')

        points = scan.ecef_pointing(t, station)*max_range
        start_point = station.ecef
        end_point = start_point[:, None] + points

        # plot local vertical vector
        if plot_local_normal is True:
            normal = frames.enu_to_ecef(station.lat, station.lon, 0, [0, 0, 1])*max_range + start_point
            ax.plot([start_point[0], normal[0]], [start_point[1], normal[1]], [start_point[2], normal[2]], "--b")

        for i in range(len(t)):            
            ax.plot([start_point[0], end_point[0, i]], [start_point[1], end_point[1, i]], [start_point[2], end_point[2, i]], "-r")
    else: # if no station is provided, plot scan in local ENU coordinates
        points = scan.enu_pointing(t, station)*max_range

        # plot local vertical vector
        if plot_local_normal is True:
            ax.plot([0, 0], [0, 0], [0, max_range], "--b")

        for i in range(len(t)):            
            ax.plot([0, points[0, i]], [0, points[1, i]], [0, points[2, i]], "-r")

    return fig, ax

def plot_radar_scan_movie(SC, earth=False, rotate=False, save_str=''):
    '''Create a animation of the scan pattern based on the :code:`_scan_time` and the :code:`_function_data['dwell_time']` variable.
    
        :param RadarScan SC: Scan to plot.
        :param bool earth: Plot the surface of the Earth.
        :param str save_str: String of path to output movie file. Requers an avalible ffmpeg encoder on the system. If string is empty no movie is saved.
    '''
    raise NotImplementedError('')
    
    if 'dwell_time' in SC._function_data:
        dwell_time = np.min(SC._function_data['dwell_time'])
    else:
        dwell_time = 0.05
    
    if SC._scan_time is None:
        scan_time = dwell_time*100.0
    else:
        scan_time = SC._scan_time
    
    t = np.linspace(0.0, scan_time, num=np.round(2*scan_time/dwell_time))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)

    def update_text(SC,t):
        return SC.name + ', t=%.4f s' % (t*1e0,)

    titl = fig.text(0.5, 0.94, update_text(SC, t[0]), size=22, horizontalalignment='center')

    max_range = 4000e3

    p0, k0= SC.antenna_pointing(0)
    p1 = p0 + k0*max_range*0.8

    if earth:
        plothelp.draw_earth_grid(ax)
    else:
        plothelp.draw_earth(ax)
        
    plothelp.draw_radar(ax, SC._lat, SC._lon)
    
    if k0[2] < 0:
        beam = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], alpha=0.5, color="red")
    else:
        beam = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], alpha=0.5, color="green")

    ax.set_xlim(p0[0] - max_range, p0[0] + max_range)
    ax.set_ylim(p0[1] - max_range, p0[1] + max_range)
    ax.set_zlim(p0[2] - max_range, p0[2] + max_range)

    interval = scan_time*1e3/float(len(t))
    rotations = np.linspace(0.,360.*2, num=len(t)) % 360.0
    
    def update(ti,beam):
        _t = t[ti]
        p0, k0= SC.antenna_pointing(_t)
        p1 = p0 + k0*max_range*0.8
        titl.set_text(update_text(SC,_t))
        beam.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        beam.set_3d_properties([p0[2], p1[2]])
        
        if k0[2] < 0:
            beam.set_color("red")
        else:
            beam.set_color("green")
        
        if rotate:
            ax.view_init(15, rotations[ti])
        
        return beam,
    
    ani = animation.FuncAnimation(fig, update, frames=range(len(t)), fargs=(beam), interval=interval, blit=False)

    if len(save_str)>0:

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(metadata=dict(artist='Daniel Kastinen'), bitrate=1800)
        ani.save(save_str, writer=writer)

    plt.tight_layout()
    plt.show()

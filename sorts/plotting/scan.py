#!/usr/bin/env python

'''Plotting helper functions

'''

#Python standard import


#Third party import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


#Local import



def radar_scan(SC, earth=False, ax = None):
    '''Plot a full cycle of the scan pattern based on the :code:`_scan_time` and the :code:`_function_data['dwell_time']` variable.
    
        :param RadarScan SC: Scan to plot.
        :param bool earth: Plot the surface of the Earth.
    '''
    raise NotImplementedError('')

    if 'dwell_time' in SC._function_data:
        dwell_time = n.min(SC._function_data['dwell_time'])
    else:
        dwell_time = 0.05
    
    if SC._scan_time is None:
        scan_time = dwell_time*100.0
    else:
        scan_time = SC._scan_time
    
    t=n.linspace(0.0, scan_time, num=n.round(2*scan_time/dwell_time))

    if ax is None:
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        ax.view_init(15, 5)
            
        plt.title(SC.name)
        plt.tight_layout()
        _figs = (fig, ax)
    else:
        _figs = (None, ax)

    if earth:
        plothelp.draw_earth_grid(ax)
    plothelp.draw_radar(ax,SC._lat,SC._lon)
    
    max_range=4000e3
    
    for i in range(len(t)):
        p0,k0=SC.antenna_pointing(t[i])
        
        p1=p0+k0*max_range*0.8
        if k0[2] < 0:
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="red")
        else:
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="green")
    
    ax.set_xlim(p0[0] - max_range, p0[0] + max_range)
    ax.set_ylim(p0[1] - max_range, p0[1] + max_range)
    ax.set_zlim(p0[2] - max_range, p0[2] + max_range)


    return _figs

def plot_radar_scan_movie(SC, earth=False, rotate=False, save_str=''):
    '''Create a animation of the scan pattern based on the :code:`_scan_time` and the :code:`_function_data['dwell_time']` variable.
    
        :param RadarScan SC: Scan to plot.
        :param bool earth: Plot the surface of the Earth.
        :param str save_str: String of path to output movie file. Requers an avalible ffmpeg encoder on the system. If string is empty no movie is saved.
    '''
    raise NotImplementedError('')
    
    if 'dwell_time' in SC._function_data:
        dwell_time = n.min(SC._function_data['dwell_time'])
    else:
        dwell_time = 0.05
    
    if SC._scan_time is None:
        scan_time = dwell_time*100.0
    else:
        scan_time = SC._scan_time
    
    t=n.linspace(0.0, scan_time, num=n.round(2*scan_time/dwell_time))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)

    def update_text(SC,t):
        return SC.name + ', t=%.4f s' % (t*1e0,)

    titl = fig.text(0.5,0.94,update_text(SC,t[0]),size=22,horizontalalignment='center')


    max_range=4000e3

    p0,k0=SC.antenna_pointing(0)
    p1=p0+k0*max_range*0.8

    if earth:
        plothelp.draw_earth_grid(ax)
    else:
        plothelp.draw_earth(ax)
    plothelp.draw_radar(ax,SC._lat,SC._lon)
    if k0[2] < 0:
        beam = ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="red")
    else:
        beam = ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="green")

    ax.set_xlim(p0[0] - max_range, p0[0] + max_range)
    ax.set_ylim(p0[1] - max_range, p0[1] + max_range)
    ax.set_zlim(p0[2] - max_range, p0[2] + max_range)

    interval = scan_time*1e3/float(len(t))
    rotations = n.linspace(0.,360.*2, num=len(t)) % 360.0
    
    def update(ti,beam):
        _t = t[ti]
        p0,k0=SC.antenna_pointing(_t)
        p1=p0+k0*max_range*0.8
        titl.set_text(update_text(SC,_t))
        beam.set_data([p0[0],p1[0]],[p0[1],p1[1]])
        beam.set_3d_properties([p0[2],p1[2]])
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

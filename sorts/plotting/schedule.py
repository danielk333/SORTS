#!/usr/bin/env python

'''Radar scan plot functions

'''

#Python standard import


#Third party import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


#Local import
from . import general
from ..controller import Tracker


def schedule_pointing(
        scheduler, 
        t0, 
        t1, 
        earth=True, 
        earth_opts={}, 
        alpha=0.1, 
        point_range=300e3, 
        view_range=300e3, 
        plot_rx = True,
        plot_tx = True,
        ax=None,
    ):
    '''Plot the schedule generated by a `PointingSchedule` instance.
    
        :param bool earth: Plot the surface of the Earth.
    '''

    if ax is None:
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
    else:
        fig = None

    if earth:
        ax = general.grid_earth(ax, **earth_opts)

    for tx in scheduler.radar.tx:
        ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]], 'or')
    for rx in scheduler.radar.rx:
        ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]], 'og')

    #get sched data
    ctrls = scheduler.get_controllers()
    times = np.concatenate([c.t[np.logical_and(c.t >= t0, c.t <= t1)] for c in ctrls], axis=0)
    sched = scheduler.chain_generators([c(c.t[np.logical_and(c.t >= t0, c.t <= t1)] - c.t0) for c in ctrls])
    sched_data = scheduler.generate_schedule(times, sched)

    for ti, t in enumerate(sched_data['t']):

        if plot_rx:
            for p, ecef in zip(sched_data['rx'][ti], sched_data['rx_pos'][ti]):
                if len(p.shape) == 1: p.shape = (3,1)

                for pi in range(p.shape[1]):
                    if 'id' in sched_data['meta'][ti]:
                        ctrl = ctrls[sched_data['meta'][ti]['id']]
                        if isinstance(ctrl, Tracker):
                            t_ind = np.argmin(np.abs(t - ctrl.t))
                            targetx = ctrl.ecefs[0,t_ind]
                            targety = ctrl.ecefs[1,t_ind]
                            targetz = ctrl.ecefs[2,t_ind]
                        else:
                            targetx = ecef[0]+p[0,pi]*point_range
                            targety = ecef[1]+p[1,pi]*point_range
                            targetz = ecef[2]+p[2,pi]*point_range
                    else:
                        targetx = ecef[0]+p[0,pi]*point_range
                        targety = ecef[1]+p[1,pi]*point_range
                        targetz = ecef[2]+p[2,pi]*point_range
                    
                    ax.plot([ecef[0], targetx], [ecef[1], targety], [ecef[2], targetz], '-g', alpha=alpha)

        if plot_tx:
            for p, ecef in zip(sched_data['tx'][ti], sched_data['tx_pos'][ti]):
                if len(p.shape) == 1: p.shape = (3,1)

                for pi in range(p.shape[1]):
                    if 'id' in sched_data['meta'][ti]:
                        ctrl = ctrls[sched_data['meta'][ti]['id']]
                        if isinstance(ctrl, Tracker):
                            t_ind = np.argmin(np.abs(t - ctrl.t))
                            targetx = ctrl.ecefs[0,t_ind]
                            targety = ctrl.ecefs[1,t_ind]
                            targetz = ctrl.ecefs[2,t_ind]
                        else:
                            targetx = ecef[0]+p[0,pi]*point_range
                            targety = ecef[1]+p[1,pi]*point_range
                            targetz = ecef[2]+p[2,pi]*point_range
                    else:
                        targetx = ecef[0]+p[0,pi]*point_range
                        targety = ecef[1]+p[1,pi]*point_range
                        targetz = ecef[2]+p[2,pi]*point_range
                    
                    ax.plot([ecef[0], targetx], [ecef[1], targety], [ecef[2], targetz], '-r', alpha=alpha)


    p0 = [tx.ecef for tx in scheduler.radar.tx]
    p0 = np.array(p0).mean(axis=0)

    ax.set_xlim(p0[0] - view_range, p0[0] + view_range)
    ax.set_ylim(p0[1] - view_range, p0[1] + view_range)
    ax.set_zlim(p0[2] - view_range, p0[2] + view_range)

    if fig is not None:
        fig.tight_layout()

    return fig, ax



def controller_slices(controllers, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = None

    times = {}
    max_t = 0
    for ctrl in controllers:
        var = ctrl.__class__.__name__
        if var not in times:
            times[var] = []

        if isinstance(ctrl.t_slice, float) or isinstance(ctrl.t_slice, int):
            t_slice = np.empty(ctrl.t.shape, dtype=np.float64)
            t_slice[:] = ctrl.t_slice
        else:
            t_slice = ctrl.t_slice

        for i in range(len(ctrl.t)):
            times[var].append((ctrl.t[i], t_slice[i]))
        
        if ctrl.t.max() > max_t:
            max_t = ctrl.t.max()

    ticks = []

    for var in times:
        ticks.append(var)
        ax.broken_barh(times[var], (len(ticks)*10, 10))
        ax.plot([x0 + 0.5*x1 for x0,x1 in times[var]], [len(ticks)*10 + 5 for x in times[var]], '.')


    ax.set_ylim(5, 15 + len(ticks)*10)
    ax.set_xlim(0, max_t*1.1)
    ax.set_xlabel('Time [s]')

    ax.set_yticks(list(range(15, 15 + 10*len(ticks), 10)))
    ax.set_yticklabels(ticks)

    ax.grid(True)


    return fig, ax
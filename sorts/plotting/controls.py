#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 06:52:45 2022

@author: thomas
"""

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

from . import general
from ..common import constants
    
def plot_beam_directions(controls, radar, ax=None, logger=None, profiler=None, tx_beam=True, rx_beam=True, zoom_level=0.7):
    '''
    Usage
    -----
    Plot beam directions obtained by applying a given set of controls (generated by a radar controller). 
    This function extracts the beam directions, computes the intersecting point between radar beams (if they exist) and plots the results on a 3d axis (provided or created automatically).
    
    The current implementation only supports the plotting of one set of controls for a unique radar system per function call.
    
    Parameters
    ----------
    
    :controls [python dictionnary]: Sets of controls applied to the radar system. The controls need to possess both the "beam_direction_tx" and "beam_direction_rx" control keys to work properly.
    :radar [sorts.system.radar]: Radar instance to which the controls set is applied to
    :ax [matplotlib.pyplot.axis]: Axis insance on which the controls are plotted. If used, the axis need to be 2 dimensional. if not provided, the function will create the axis automatically.
    :logger [Logging.logger]: logger instance used to log the execution of the function
    :profiler [Profiling.profiler]: profiler instance used to compute the execution performances of the function
    :tx_beam [boolean]: if True, the function will plot the beams relative to the Tx stations. If not, the plotting of Tx stations will be skipped automatically
    :rx_beam [boolean]: if True, the function will plot the beams relative to the Rx stations. If not, the plotting of Rx stations will be skipped automatically
    :zoom_level [float]: zoom level of the final plot (has to be in [0; 1]).
    
    
    Return value
    ------------
    
    Instance of the provided or created matplotlib axis.
    
    Example
    -------
    
    The plotting of a set of beam orientation controls can be achieved in the following manner. 
    First, one need to initialize the radar instance to be controlled as well as the controller :
    
    >>> from sorts.radar.scans import Fence        # import scans
    >>> from sorts.radar.system import instances   # import the radar instances (such as eiscat3d)
    >>> from sorts.radar import controllers        # import the controllers
    >>> from sorts import equidistant_sampling 
    >>> from sorts import plotting                 # import the plotting functions

    >>> scan = Fence(azimuth=90, min_elevation=30, dwell=0.1, num=50) # Scan type definition
    >>> eiscat3d = instances.eiscat3d              # RADAR definition
    >>> scanner_ctrl = controllers.Scanner()
        
    Then, one can call the generate_controls method to generate the beam orientation controls :
    
    >>> t = np.arange(0, end_t, scan.dwell())
    >>> controls = scanner_ctrl.generate_controls(t, eiscat3d, scan, priority=-1)
        
    Finally, the results can be plotted by calling :
    
    >>> plotting.plot_beam_directions(controls["beam_direction"], eiscat3d, logger=logger)
    
    '''
    # Validate inputs
    if ax is None or not hasattr(ax, 'get_zlim'):
        if logger is not None:
            logger.info("plotting:controls:plot_beam_directions -> No valid plotting axis provided, generating new figure")
        
        fig = plt.figure(dpi=300, figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        
    if not "tx" in controls.keys():
        raise KeyError("No Tx beam direction controls found in the provided controls.")
    
    if not "rx" in controls.keys():
        raise KeyError("No Rx beam direction controls found in the provided controls.")
    
    # Starting execution
    if profiler is not None:
        profiler.start("plotting:controls:plot_beam_directions")
        profiler.start("plotting:controls:plot_beam_directions:plot_stations")
    
    # Plotting station ECEF positions
    general.grid_earth(ax)
    
    # Plotting station ECEF positions
    for tx in radar.tx:
        ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
    for rx in radar.rx:
        ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')
    
    if profiler is not None:
        profiler.stop("plotting:controls:plot_beam_directions:plot_stations")
        profiler.start("plotting:controls:plot_beam_directions:plot_beams")
        
    # get Tx station positions
    tx_ecef = np.array([tx.ecef for tx in radar.tx], dtype=float)
    rx_ecef = np.array([rx.ecef for rx in radar.rx], dtype=float)
    
    # compute point
    rx_directions = np.reshape(controls["rx"].transpose(0, 3, 1, 4, 2), (np.shape(rx_ecef)[0], 3, -1), 'C') # convert array to get [x, y, z] coordinates
    tx_directions = np.reshape(controls["tx"], (np.shape(tx_ecef)[0], 3, -1), 'C').repeat(np.shape(controls["rx"])[2], axis=2) # convert array to get [x, y, z] coordinates
    
    # generate points (ecef frame)
    a = np.einsum('ijk,hjk->ihk', tx_directions, rx_directions)
    
    msk = np.abs(np.abs(a) - 1) > 0.0001
    
    # compute points for Tx stations [[x, y, z], station, points]
    for txi in range(len(radar.tx)):
        for rxi in range(len(radar.rx)):
            if len(a[txi, rxi, msk[txi, rxi, :]]) > 0:
                if profiler is not None:
                    profiler.start("plotting:controls:plot_beam_directions:plot_beams:compute_intersection_points")
                    
                # check if vectors are in the same plane (i.e. if the radar beams intersect)
                k_tx_rx = tx_ecef[None, txi, :] - rx_ecef[None, rxi, :]
                k_tx_rx = k_tx_rx/np.linalg.norm(k_tx_rx)
                
                M = np.array([np.repeat(k_tx_rx, np.shape(tx_directions[txi, :, msk[txi, rxi, :]])[0], axis=0), tx_directions[txi, :, msk[txi, rxi, :]], rx_directions[rxi, :, msk[txi, rxi, :]]], dtype=float).transpose(1, 0, 2)
                mask = np.abs(np.linalg.det(M)) < 1e-5
                
                # computing intersection point between a given rx/tx station tuple beams
                points = tx_ecef[None, txi, :].transpose() - np.einsum("ij,i->ji", tx_directions[txi, :, msk[txi, rxi, :]][mask], np.dot(rx_directions[rxi, :, msk[txi, rxi, :]][mask]*a[txi, rxi, :, None][mask] - tx_directions[txi, :, msk[txi, rxi, :]][mask], tx_ecef[txi, :] - rx_ecef[rxi, :])/(a[txi, rxi, :][mask]**2 - 1))
                
                if profiler is not None:
                    profiler.stop("plotting:controls:plot_beam_directions:plot_beams:compute_intersection_points")
                    profiler.start("plotting:controls:plot_beam_directions:plot_beams:plot_rx")
                    
                # plot Rx station beams
                if rx_beam is True:
                    points_rx = np.tile(points.repeat(2, axis=1), np.shape(rx_ecef)[0])
                    points_rx[:, ::2] = np.repeat(rx_ecef.transpose().reshape(3, -1), points.shape[-1], axis=1)
                    
                    ax.plot(points_rx[0], points_rx[1], points_rx[2], 'g-', alpha=0.5, linewidth=0.15)
                else:
                    if logger is not None:
                        logger.info("plotting:controls:plot_beam_directions:plot_station_controls:rx:{rxi} -> rx_beam is False, skipping rx controls plotting...")
                
                if profiler is not None:
                    profiler.stop("plotting:controls:plot_beam_directions:plot_beams:plot_rx")
                    profiler.start("plotting:controls:plot_beam_directions:plot_beams:plot_tx")
                    
                # plot Tx station beams 
                if tx_beam is True:
                    points_tx = points.repeat(2, axis=1)
                    points_tx[:, ::2] = np.repeat(tx_ecef.transpose().reshape(3, -1), points.shape[-1], axis=1)
                    
                    ax.plot(points_tx[0], points_tx[1], points_tx[2], 'r-', alpha=0.5, linewidth=0.5)
                else:
                    if logger is not None:
                        logger.info("plotting:controls:plot_beam_directions:plot_station_controls:tx:{txi} -> x_beam is False, skipping tx controls plotting...")
                
                if profiler is not None:
                    profiler.stop("plotting:controls:plot_beam_directions:plot_beams:plot_tx")
                    
    # Zooming view on the stations performing the controls
    if zoom_level is not None: 
        if zoom_level > 1 or zoom_level < 0: 
            zoom_level = 0;
            
            if logger is not None:
                logger.info("plotting:controls:zoom -> zoom level is invalid (outside of bounds [0; 1]). Using zoom level 0%")
        else:
            # min/max zoom levels 
            dr_min = 100e3
            dr_max = 1.5 * constants.R_earth
            
            # comnputing zoom level between zmin and zmax
            dr = (dr_min - dr_max)*zoom_level + dr_max
            
            # set zoom level
            ax.set_xlim([radar.tx[0].ecef[0] - dr, radar.tx[0].ecef[0] + dr])
            ax.set_ylim([radar.tx[0].ecef[1] - dr, radar.tx[0].ecef[1] + dr])
            ax.set_zlim([radar.tx[0].ecef[2] - dr, radar.tx[0].ecef[2] + dr])
    
    if profiler is not None:    
        profiler.stop("plotting:controls:plot_beam_directions")
    
    return ax # return instance of axis for further use if necessary
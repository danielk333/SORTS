#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 06:52:45 2022

@author: thomas
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
import ctypes

import matplotlib.pyplot as plt
import numpy as np

from . import general
from ..common import constants
from ..radar import controls_manager

from sorts import clibsorts

MAX_SUB_PLOTS = 30

def plot_beam_directions(controls, radar, ax=None, logger=None, profiler=None, tx_beam=True, rx_beam=True, zoom_level=0.95, azimuth=45, elevation=45, alpha=1, linewidth_tx=0.5, linewidth_rx=0.5, fmt=None):
    '''
    Usage
    -----
    Plot beam directions obtained by applying a given set of controls (generated by a radar controller). 
    This function extracts the beam directions, computes the intersecting point between radar beams (if they exist) and plots the results on a 3d axis (provided or created automatically).
    
    The current implementation only supports the plotting of one set of controls for a unique radar system per function call.
    
    Parameters
    ----------
    
    :controls [dict]: Sets of controls applied to the radar system. The controls need to possess both the "beam_direction_tx" and "beam_direction_rx" control keys to work properly.
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
    
    if profiler is not None:
        profiler.stop("plotting:controls:plot_beam_directions:plot_stations")
        profiler.start("plotting:controls:plot_beam_directions:plot_beams")
        
    # get Tx station positions
    tx_ecef = np.array([tx.ecef for tx in radar.tx], dtype=float)
    rx_ecef = np.array([rx.ecef for rx in radar.rx], dtype=float)

    if not isinstance(controls["rx"], np.ndarray):
        controls["rx"] = np.asarray(controls["rx"])

    if not isinstance(controls["tx"], np.ndarray):
        controls["tx"] = np.asarray(controls["tx"])

    if controls["rx"].dtype == object:
        tx_directions, rx_directions = __flatten_orientation_array(controls)
    else:
        # compute point -> [rx, tx, time, t_slice, (x,y,z)] -
        rx_directions = np.reshape(np.asfarray(controls["rx"].transpose(0, 1, 4, 2, 3)), (np.shape(rx_ecef)[0], np.shape(tx_ecef)[0], 3, -1), 'C') # convert array to get [x, y, z] coordinates
        
        n_repeats = int(np.shape(controls["rx"])[3]/np.shape(controls["tx"])[3])
        tx_directions = np.reshape(np.asfarray(controls["tx"].transpose(0, 1, 4, 2, 3)), (np.shape(tx_ecef)[0], 1, 3, -1), 'C').repeat(n_repeats, axis=3) # convert array to get [x, y, z] coordinates  

    # generate points (ecef frame) [tx, rx, points]
    a = np.einsum('ijkl,hjkl->ihl', tx_directions, rx_directions)
    
    msk = np.abs(np.abs(a) - 1) > 0.0001
    
    # compute points for Tx stations [[x, y, z], station, points]
    for txi in range(len(radar.tx)):
        for rxi in range(len(radar.rx)):
            if len(a[txi, rxi, msk[txi, rxi, :]]) > 0:
                if profiler is not None:
                    profiler.start("plotting:controls:plot_beam_directions:plot_beams:compute_intersection_points")
                    
                # check if vectors are in the same plane (i.e. if the radar beams intersect)
                k_tx_rx = tx_ecef[None, txi, :] - rx_ecef[None, rxi, :]
                k_norm = np.linalg.norm(k_tx_rx)
                
                if k_norm > 0:
                    k_tx_rx = k_tx_rx/k_norm
                    
                    M = np.array([np.repeat(k_tx_rx, np.shape(tx_directions[txi, 0, :, msk[txi, rxi, :]])[0], axis=0), tx_directions[txi, 0, :, msk[txi, rxi, :]], rx_directions[rxi, txi, :, msk[txi, rxi, :]]], dtype=float).transpose(1, 0, 2)
                    mask = np.abs(np.linalg.det(M)) < 1e-5 # if det = 0 then the directions are in the same plane
    
                    # TODO : repare this part
                    # computing intersection point between a given rx/tx station tuple beams
                    if np.size(np.where(mask == True)[0]) > 0: 
                        points = tx_ecef[txi, None].reshape(3, 1) - np.einsum("ij,i->ji", tx_directions[txi, 0, :,  msk[txi, rxi, :]][mask], np.dot(rx_directions[rxi, txi, :, msk[txi, rxi, :]][mask]*a[txi, rxi, msk[txi, rxi, :], None][mask] - tx_directions[txi, 0, :, msk[txi, rxi, :]][mask], tx_ecef[txi, :] - rx_ecef[rxi, :])/(a[txi, rxi, msk[txi, rxi, :]][mask]**2 - 1))
                        
                        if profiler is not None:
                            profiler.stop("plotting:controls:plot_beam_directions:plot_beams:compute_intersection_points")
                            profiler.start("plotting:controls:plot_beam_directions:plot_beams:plot_rx")
                            
                        # plot Rx station beams
                        if rx_beam is True:
                            points_rx = np.tile(points.repeat(2, axis=1), np.shape(rx_ecef)[0])
                            points_rx[:, ::2] = np.repeat(rx_ecef.transpose().reshape(3, -1), points.shape[-1], axis=1)
                            
                            fmt_rx = "g-"
                            if fmt is not None:
                                fmt_rx = fmt

                            ax.plot(points_rx[0], points_rx[1], points_rx[2], fmt_rx, alpha=0.5, linewidth=linewidth_rx)
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
                            
                            fmt_tx = "r-"
                            if fmt is not None:
                                fmt_tx = fmt
                                
                            ax.plot(points_tx[0], points_tx[1], points_tx[2], fmt_tx, alpha=alpha, linewidth=linewidth_tx)
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
    
    ax.view_init(elevation, azimuth)

    return ax # return instance of axis for further use if necessary


def plot_manager_control_sequence(controls, final_controls, period_indices, manager, logger=None, profiler=None):
    # Validate inputs
    if not issubclass(manager.__class__, controls_manager.RadarControlManagerBase):
        raise ValueError(f"manager must be a sub class of {controls_manager.RadarControlManagerBase}")

    if len(controls) > MAX_SUB_PLOTS:
        raise ValueError(f"too many controls to plot ({len(controls)} > {MAX_SUB_PLOTS})")

    if np.size(period_indices) == 1:
        period_indices = np.array([period_indices], dtype=np.int32)
    else:
        period_indices = np.asarray(period_indices, dtype=np.int32)

    figs = []
    for period_index in period_indices:
        t_start = manager.t0 + period_index * manager.manager_period
        t_end = t_start + manager.manager_period

        control_indices = []
        control_period_indices = []

        for ctrl_id in range(len(controls)):
            if not "t" in controls[ctrl_id].keys():
                raise KeyError("No time array found in controls")

            if period_index == len(period_indices)-1:
                real_control_end_time = controls[ctrl_id]["t"][-1][-1] + controls[ctrl_id]["t_slice"][-1][-1]

                if t_end < real_control_end_time:
                    t_end = real_control_end_time

            if controls[ctrl_id]["t"][-1][-1] < t_start or controls[ctrl_id]["t"][0][0] >= t_start + manager.manager_period:
                if logger is not None:
                    logger.info(f"plotting:controls:plot_manager_control_sequence -> controls id={ctrl_id} : no controls in manager period [{t_start}, {t_start+manager.manager_period}]")
            else:
                for control_period_id in range(len(controls[ctrl_id]["t"])):
                    if controls[ctrl_id]["t"][control_period_id][0] >= t_start and controls[ctrl_id]["t"][control_period_id][0] < t_start + manager.manager_period:
                        control_indices.append(ctrl_id)
                        control_period_indices.append(control_period_id)

        if len(control_indices) == 0:
            if logger is not None:
                logger.info(f"plotting:controls:plot_manager_control_sequence -> no controls found in manager period [{t_start}, {t_start+manager.manager_period}]")
        else:
            fig = plt.figure(figsize=(5, 5))
            axes = fig.subplots(len(controls)+1, 1, sharex=True)
            figs.append(fig)

            for axi, ax in enumerate(axes):
                ax.set_ylabel("Control #" + str(axi))

                ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
                ax.yaxis.set_minor_locator(AutoMinorLocator(n=10))
                ax.xaxis.set_minor_locator(AutoMinorLocator(n=10))

                ax.grid(axis="x", which="both")

            axes[-1].set_ylabel("Final sequence")
            axes[-1].set_xlabel("t [s]")

            for ctrl_id in range(len(controls)):
                id_ = np.where(np.array(control_indices) == ctrl_id)[0]

                if len(id_) > 0:     
                    __plot_control_uptime(controls[ctrl_id], control_period_indices[id_[0]], t_start, t_end, axes[ctrl_id])

            # plot final controls
            __plot_control_uptime(final_controls, period_index, t_start, t_end, axes[-1])
            __plot_active_control_uptime(final_controls, period_index, t_start, t_end, axes)

            for ax in axes:
                ax.set_xlim([t_start, t_end])
                ax.set_ylim([-0.5, 1.5])

    return figs


def __plot_control_uptime(control, control_period_id, t_start, t_end, ax):
    # transform the time array to plot the status of the control at a given time t
    ctrl_t = np.repeat(control["t"][control_period_id], 4)
    ctrl_t[2::4] = control["t_slice"][control_period_id] + control["t"][control_period_id]
    ctrl_t[3::4] = control["t_slice"][control_period_id] + control["t"][control_period_id]

    ctrl_status = np.tile(np.array([0, 1, 1, 0]), int(np.size(ctrl_t)/4))

    ctrl_t, ctrl_status = boundary(control, control_period_id, t_start, t_end, ctrl_t, ctrl_status)

    # plot control array
    ax.plot(ctrl_t, ctrl_status, "-b")
    
    
def boundary(control, control_period_id, t_start, t_end, ctrl_t, ctrl_status):
    dt = 0
    if control_period_id > 0:
        dt = control["t_slice"][control_period_id-1][-1] + control["t"][control_period_id-1][-1] - t_start

    if dt > 0:
        ctrl_t = np.append([t_start, t_start + dt, t_start + dt], ctrl_t)
        ctrl_status = np.append([1, 1, 0], ctrl_status)
    else:
        ctrl_t = np.append([t_start], ctrl_t)
        ctrl_status = np.append([0], ctrl_status)

    ctrl_t = np.append(ctrl_t, [t_end])
    ctrl_status = np.append(ctrl_status, [0])

    return ctrl_t, ctrl_status

def __plot_active_control_uptime(final_control_sequence, period_index, t_start, t_end, axes):
    # transform the time array to plot the status of the control at a given time t
    for i, ctrl_id in enumerate(final_control_sequence['active_control'][period_index]):
        t = np.array([final_control_sequence['t'][period_index][i], final_control_sequence['t'][period_index][i] + final_control_sequence['t_slice'][period_index][i]]).repeat(2)
        ctrl = np.array([0, 1, 1, 0])

        axes[ctrl_id].plot(t, ctrl, "-r")

    if period_index > 0:
        ctrl_id = final_control_sequence['active_control'][period_index-1][-1]
        dt = final_control_sequence['t'][period_index-1][-1] + final_control_sequence['t_slice'][period_index-1][-1] - t_start

        if dt > 0:
            axes[ctrl_id].plot([t_start, t_start + dt, t_start + dt], [1, 1, 0], "-r")


def __flatten_orientation_array(controls):
    tx_dirs = None
    rx_dirs = None

    n_tx = len(controls["tx"])
    n_rx = len(controls["rx"])

    n_time_points = len(controls["tx"][0][0])

    def get_pointing_direction(txi, rxi, ti, pdir, istx):
        nonlocal controls
        if istx == 1:
            pointing_direction = controls["tx"][txi][0][ti]
        else:
            pointing_direction = controls["rx"][rxi][txi][ti]   

        pdir_arr = np.ctypeslib.as_array(pdir, (len(pointing_direction), 3))
        pdir_arr[:] = pointing_direction

        del pointing_direction

    def get_n_dirs_per_time_slice(txi, rxi, ti, istx):
        nonlocal controls
        if istx == 1:
            return len(controls["tx"][txi][0][ti])
        else:
            return len(controls["rx"][rxi][txi][ti])

    def save_pointing_direction_arrays(tx_dirs_c, rx_dirs_c, n_dirs_tx, n_dirs_rx, dim_index):
        nonlocal tx_dirs, rx_dirs, n_tx, n_rx

        n_sub_rx = int(n_dirs_rx/n_rx)
        n_sub = int(n_sub_rx/n_tx)
        
        if tx_dirs is None:
            tx_dirs = np.ndarray((n_tx, 1, 3, n_dirs_tx), dtype=float)
        if rx_dirs is None:
            rx_dirs = np.ndarray((n_rx, n_tx, 3, n_sub), dtype=float)

        if n_dirs_tx > 0:
            # copy tx_dirs
            buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            buffer = buffer_from_memory(tx_dirs_c, np.dtype(np.float64).itemsize*n_dirs_tx)

            tmp = np.frombuffer(buffer, np.float64).astype(float)
            for i in range(n_tx):
                tx_dirs[i, 0, dim_index, :] = tmp[i*n_sub:(i+1)*n_sub]
            
        if n_dirs_rx > 0:
            # copy tx_dirs
            buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            buffer = buffer_from_memory(rx_dirs_c, np.dtype(np.float64).itemsize*n_dirs_rx)

            tmp = np.frombuffer(buffer, np.float64).astype(float)

            for i in range(n_rx):
                i_start = n_sub_rx*i
                for j in range(n_tx):
                    j_start = i_start + n_sub*j
                    rx_dirs[i, j, dim_index, :] = tmp[j_start:j_start+n_sub]


    GET_N_DIRS_PER_TIME_SLICE_FNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    GET_POINTING_DIRECTION_FNC = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
    SAVE_POINTING_ARRAYS_FNC = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int)
    
    get_n_dirs_per_time_slice_c = GET_N_DIRS_PER_TIME_SLICE_FNC(get_n_dirs_per_time_slice)
    get_pointing_direction_c = GET_POINTING_DIRECTION_FNC(get_pointing_direction)
    save_pointing_direction_arrays_c = SAVE_POINTING_ARRAYS_FNC(save_pointing_direction_arrays)

    clibsorts.init_plotting_controls.argtypes = [
        GET_POINTING_DIRECTION_FNC, 
        GET_N_DIRS_PER_TIME_SLICE_FNC, 
        SAVE_POINTING_ARRAYS_FNC,
        ]
    clibsorts.init_plotting_controls(
        get_pointing_direction_c, 
        get_n_dirs_per_time_slice_c, 
        save_pointing_direction_arrays_c,
        )

    clibsorts.flatten_directions.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    clibsorts.flatten_directions(
        ctypes.c_int(n_tx), 
        ctypes.c_int(n_rx), 
        ctypes.c_int(n_time_points),
        )
    
    return tx_dirs, rx_dirs
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 08:40:28 2022

@author: Thomas Maynadié
"""
import multiprocessing as mp

from pylab import figure, cm

import numpy as np
import ctypes
import matplotlib.pyplot as plt
import scipy as sp
import itertools
from matplotlib.ticker import AutoMinorLocator
import pyorb

from sorts import radars
from sorts import controllers
from sorts import space_object
from sorts import find_simultaneous_passes, equidistant_sampling
from sorts import plotting

from sorts.common import profiling
from sorts.common import interpolation
from sorts.targets.propagator import Kepler
from sorts.common import multiprocessing_tools as mptools

from coherent_integration import core

eiscat3d = radars.eiscat3d

def main():
    # Profiler
    p = profiling.Profiler()
    logger = profiling.get_logger('scanning')

    # get object states :
    end_t = 3600*24

    #radar properties
    f_radar = 930e6
    T_radar = 1/f_radar
    c = 3e8

    # Observation simulation
    # max values (for integration)
    R_max = 2000e3
    v_max = 50e3

    fd_max = 2*f_radar*v_max/c
    Fe_min = 2*fd_max
    Te_max = 1/Fe_min    

    # pulse properties :
    IPP = 2*R_max/c
    pulse_duration = 0.1*IPP

    N_ipp_per_time_slice = 1
    time_slice = IPP * N_ipp_per_time_slice

    t = np.arange(0, end_t, IPP)

    N_points_per_ipp = int(IPP/Te_max) # N
    
    # mixing frequency
    f_mix = f_radar
    code_signal = True

    t_states, interpolated_states, eiscat_passes = get_object_passes(time_slice, end_t, p, logger)

    # main pulse
    t_tx_pulse, tx_pulse = core.create_radar_pulse(1, core.pulse_function, f_radar, f_mix, N_points_per_ipp, IPP, pulse_duration, code=code_signal)
    
    for pass_id in range(len(eiscat_passes)):
        t_states_pass_i = t_states[eiscat_passes[pass_id].inds]

        N_slices = int((t_states_pass_i[-1] - t_states_pass_i[0])/5) # 1 tslice each second
        t_samp = np.linspace(t_states_pass_i[0], t_states_pass_i[-1], N_slices)

        t_ipp_samp = np.repeat(t_samp, N_ipp_per_time_slice)
        for i in range(N_ipp_per_time_slice):
            t_ipp_samp[i::N_ipp_per_time_slice] += i*IPP

        print(t_ipp_samp)

        # correlation properties
        corr_shared = mptools.convert_to_shared_array(np.zeros((N_points_per_ipp, len(t_samp))), ctypes.c_double)
        corr = mptools.convert_to_numpy_array(corr_shared, (N_points_per_ipp, len(t_samp)))

        del t_states_pass_i

        # get actual position/range of the object    
        r = np.linalg.norm(interpolated_states.get_state(t_ipp_samp)[0:3, :] - eiscat3d.tx[0].ecef[:, None], axis=0)
        print(r)

        # target properties  
        t_shift = 2*r/c
        v = interpolated_states.get_state(t_ipp_samp)[3:6]
        f_doppler = 2*np.einsum("ij,ij->j", v, interpolated_states.get_state(t_ipp_samp)[0:3, :]/r)/c*f_radar

        t_echo = np.linspace(0, IPP, N_points_per_ipp)

        def f(ti, corr_shared, mutex):
            nonlocal t_echo, N_ipp_per_time_slice, t_shift, f_doppler, f_mix, pulse_duration, code_signal, N_points_per_ipp, IPP, t_samp
            print(f"generating echo at T_s={t_shift[ti]} and fd={f_doppler[ti]}")

            radar_echo = np.empty(N_ipp_per_time_slice*len(t_echo), dtype=complex)
            reference = np.empty(N_ipp_per_time_slice*len(t_echo), dtype=complex)

            for ipp in range(N_ipp_per_time_slice):
                radar_echo[ipp*len(t_echo):(ipp+1)*len(t_echo)] = core.create_echo(t_echo, 0.5, f_radar, f_mix, f_doppler[ti+ipp], t_shift[ti+ipp], pulse_duration, code=code_signal)
                radar_echo[ipp*len(t_echo):(ipp+1)*len(t_echo)] += np.random.normal(0, 0.4, len(t_echo))

            t_ref_tx_pulse, ref_tx_pulse = core.create_radar_pulse(1, core.pulse_function, f_radar + f_doppler[ti+ipp], f_mix, N_points_per_ipp, IPP, pulse_duration, code=code_signal)
            reference = ref_tx_pulse

            mutex.acquire()
            corr = mptools.convert_to_numpy_array(corr_shared, (N_points_per_ipp, len(t_samp)))
            corr[:, ti] = core.correlate(radar_echo, reference, N_ipp_per_time_slice, 1)
            mutex.release()

            print(f"iteration {ti} over {len(t_samp)}")
            del radar_echo

        max_processes = 20
        for process_subgroup_id in range(int(len(t_samp)/max_processes) + 1):
            if int(len(t_samp) - process_subgroup_id*max_processes) >= max_processes:
                n_process_in_subgroup = max_processes
            else:
                n_process_in_subgroup = int(len(t_samp) - process_subgroup_id*max_processes)

            mutex = mp.Lock() # create the mp.Lock mutex to ensure critical ressources sync between processes
            process_subgroup = []

            # initializes each process and associate them to an object in the list of targets to follow
            for i in range(n_process_in_subgroup):
                ti = process_subgroup_id * max_processes + i # get the object's id

                process = mp.Process(target=f, args=(ti, corr_shared, mutex,)) # create new process
                process_subgroup.append(process)
                process.start()

            # wait for each process to be finished
            for process in process_subgroup:
                process.join()

        fig = plt.figure(figsize=(1, 0.5), dpi=300)
        ax1 = fig.add_subplot(121, projection='3d')

        plotting.grid_earth(ax1, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

        # Plotting station ECEF positions
        logger.info("test_tracking_scheduler -> plotting radar stations")
        for tx in eiscat3d.tx:
            ax1.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
        for rx in eiscat3d.rx:
            ax1.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')    

        plot_controls(t_samp, interpolated_states.get_state(t_samp), time_slice, N_ipp_per_time_slice, ax1)
        ax1.plot(interpolated_states.get_state(t_samp)[0], interpolated_states.get_state(t_samp)[1], interpolated_states.get_state(t_samp)[2], "-b")

        ax2 = fig.add_subplot(122)
        r_samp_km = r.reshape(N_slices, -1)[:,0].flatten()/1000
        r_max_km = R_max/1000

        im = ax2.imshow(corr, aspect='auto', cmap=cm.jet, extent=[t_ipp_samp[0], t_ipp_samp[-1], 0, r_max_km], origin='lower')
        cbar = fig.colorbar(im)
        cbar.set_label('Range signal correlation', rotation=270, labelpad=20)

        # ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
        # ax.xaxis.set_minor_locator(AutoMinorLocator())


        ax2.set_ylabel(r"Range (Tx) [km]")
        ax2.set_xlabel(r"time [s]")

        ax2.plot(t_samp, r_samp_km, "--r", linewidth=1)

        ax2.set_ylim([0.9*min(r_samp_km), min([1.1*max(r_samp_km), r_max_km])])

        plt.show()


def get_object_passes(t_slice, end_t, p, logger, max_points=100):
    # RADAR definition
    logger.info(f"test_tracker_controller -> initializing radar insance eiscat3d={eiscat3d}")

    # Object definition
    # Propagator
    Prop_cls = Kepler
    Prop_opts = dict(
        settings = dict(
            out_frame='ITRS',
            in_frame='TEME',
        ),
    )
    logger.info(f"test_tracker_controller -> initializing propagator ({Kepler}) options ({Prop_opts})")


    # Creating space object
    # Object properties
    orbits_a = np.array([7000, 8500, 7500, 10000])*1e3 # km
    orbits_i = np.array([80, 105, 105, 80]) # deg
    orbits_raan = np.array([86, 160, 180, 90]) # deg
    orbits_aop = np.array([0, 50, 40, 55]) # deg
    orbits_mu0 = np.array([50, 5, 30, 8]) # deg
    obj_id = 0

    p.start('Total')
    p.start('object_initialization')

    # Object instanciation
    logger.info("test_tracker_controller -> creating new object\n")

    target = space_object.SpaceObject(
            Prop_cls,
            propagator_options = Prop_opts,
            a = orbits_a[obj_id], 
            e = 0.05,
            i = orbits_i[obj_id],
            raan = orbits_raan[obj_id],
            aop = orbits_aop[obj_id],
            mu0 = orbits_mu0[obj_id],
            
            epoch = 53005.0,
            parameters = dict(
                d = 0.1,
            ),
        )

    p.stop('object_initialization')
    logger.info("test_tracker_controller -> object created :")
    logger.info(f"test_tracker_controller -> {target}")

    logger.info("test_tracker_controller -> sampling equidistant states on the orbit")

    # create state time array
    p.start('equidistant_sampling')
    t_states = np.arange(0, end_t, 30.0)
    p.stop('equidistant_sampling')

    logger.info(f"test_tracker_controller -> sampling done : t_states -> {t_states.shape}")


    # get object states in ECEF frame
    p.start('get_state')
    object_states = target.get_state(t_states)
    p.stop('get_state')

    logger.info(f"test_tracker_controller -> object states computation done ! ")
    logger.info(f"test_tracker_controller -> t_states -> {t_states.shape}")


    # reduce state array
    p.start('find_simultaneous_passes')
    eiscat_passes = find_simultaneous_passes(t_states, object_states, [*eiscat3d.tx, *eiscat3d.rx])
    p.stop('find_simultaneous_passes')
    logger.info(f"test_tracker_controller -> Passes : eiscat_passes={eiscat_passes}")

    interpolated_states = interpolation.Legendre8(object_states, t_states)
    del object_states

    return t_states, interpolated_states, eiscat_passes

def plot_controls(time_points, target_ecef, t_slice, states_per_slice, ax):
    tracker_controller = controllers.Tracker()
    controls = tracker_controller.generate_controls(time_points, eiscat3d, time_points, target_ecef, t_slice=t_slice, max_points=1000, priority=0, states_per_slice=states_per_slice)

    for period_id in range(len(controls["t"])):
        ax = plotting.plot_beam_directions(next(controls["pointing_direction"]), eiscat3d, ax=ax, zoom_level=0.6, azimuth=10, elevation=20)

if __name__ == "__main__":
    main()
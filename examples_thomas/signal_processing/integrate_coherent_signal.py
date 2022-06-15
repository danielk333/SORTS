# -*- coding: utf-8 -*-
"""
Created on Sat May  7 08:40:28 2022

@author: Thomas Maynadié
"""

from pylab import figure, cm

import numpy as np
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

from coherent_integration import core

def main():
    # Profiler
    p = profiling.Profiler()
    logger = profiling.get_logger('scanning')

    # get object states :
    end_t = 100

    #radar properties
    f_radar = 930e6
    T_radar = 1/f_radar
    c = 3e8

    # Observation simulation
    # max values (for integration)
    R_max = 1000e3
    v_max = 10e3

    fd_max = 2*f_radar*v_max/c
    Fe_min = 2*fd_max
    Te_max = 1/Fe_min    

    # pulse properties :
    IPP = 0.2#2*R_max/c
    pulse_duration = 0.1*IPP

    N_ipp_per_time_slice = 5
    N_slices = 10
    time_slice = IPP * N_ipp_per_time_slice
    t = np.arange(0, end_t, IPP)

    N_points_per_ipp = int(IPP/Te_max) # N
    
    # mixing frequency
    f_mix = f_radar
    code_signal = True

    t_states, object_states, passes = get_object_passes(time_slice, end_t, p, logger)

    # main pulse
    t_tx_pulse, tx_pulse = core.create_radar_pulse(1, core.pulse_function, f_radar, f_mix, N_points_per_ipp, IPP, pulse_duration, code=code_signal)
    t_samp = t.reshape(N_slices, -1)[:,0].flatten()
    t_ipp_samp = t.reshape(N_slices, -1)[:,0:N_ipp_per_time_slice].flatten()

    print(t_ipp_samp)
    print(t_samp)

    # correlation properties
    corr = np.zeros((N_points_per_ipp, len(t_samp)))

    for pass_id in range(len(passes)):
        t_states_pass_i = t_states[passes[pass_id].inds]
        tracking_states = object_states[:, passes[pass_id].inds]

        interpolated_states = interpolation.Linear(tracking_states, t_states_pass_i)
        del t_states_pass_i, tracking_states

        # get actual position/range of the object    
        r = np.linalg.norm(interpolated_states.get_state(t_ipp_samp)[0:3, :], axis=0)

        # target properties  
        t_shift = 2*r/c
        v = interpolated_states.get_state(t_ipp_samp)[3:6]
        f_doppler = 2*np.einsum("ij,ij->j", v, interpolated_states.get_state(t_ipp_samp)[0:3, :]/r)/c*f_radar

        t_echo = np.linspace(0, IPP, N_points_per_ipp)

        for ti, t_ in enumerate(t_samp):
            print(f"generating echo at T_s={t_shift[ti]} and fd={f_doppler[ti]}")

            radar_echo = np.empty(N_ipp_per_time_slice*len(t_echo), dtype=complex)
            reference = np.empty(N_ipp_per_time_slice*len(t_echo), dtype=complex)

            for ipp in range(N_ipp_per_time_slice):
                radar_echo[ipp*len(t_echo):(ipp+1)*len(t_echo)] = core.create_echo(t_echo, 0.75, f_radar, f_mix, f_doppler[ti+ipp], t_shift[ti+ipp], pulse_duration, code=code_signal)
                radar_echo[ipp*len(t_echo):(ipp+1)*len(t_echo)] += np.random.normal(0, 0.4, len(t_echo))

                # for i in range(5):
                #     f_echo_noise = np.random.normal(0, 0.5, 1)*20
                #     t_s_noise = np.random.normal(IPP/2, 0.5, 1)*IPP/4

                #     radar_echo[ipp*len(t_echo):(ipp+1)*len(t_echo)] += core.create_echo(t_echo, 0.1, f_radar, f_mix, f_echo_noise, t_s_noise, pulse_duration, code=code_signal)

                t_ref_tx_pulse, ref_tx_pulse = core.create_radar_pulse(1, core.pulse_function, f_radar + f_doppler[ti+ipp], f_mix, N_points_per_ipp, IPP, pulse_duration, code=code_signal)
                reference[ipp*len(t_echo):(ipp+1)*len(t_echo)] = ref_tx_pulse

            corr[:, ti] = core.correlate(radar_echo, reference, N_ipp_per_time_slice, 1)
            
            print(f"iteration {ti} over {len(t_samp)}")
            #print(radar_echo)

            del radar_echo
    
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)

        r_samp = r.reshape(N_slices, -1)[:,0].flatten()

        r_min = min(r_samp)*0.75
        r_max = max(r_samp)*1.55

        i_r_min = int(r_min*2/c/IPP*N_points_per_ipp)
        i_r_max = int(r_max*2/c/IPP*N_points_per_ipp)

        im = ax.imshow(corr[i_r_min:i_r_max], aspect='auto', cmap=cm.jet, extent=[t_samp[0], t_samp[-1], r_min, r_max], origin='lower')
        fig.colorbar(im)

        # ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
        # ax.xaxis.set_minor_locator(AutoMinorLocator())

        ax.set_ylabel(r"Range [m]")
        ax.set_xlabel(r"time [s]")

        ax.plot(t_samp, r_samp, "--r", linewidth=1)

        plt.show()
    
    # # get velocity and range slices
    # imax, jmax = np.unravel_index(np.argmax(autocorr), autocorr.shape)
    
    # fig = plt.figure(dpi=300)
    # fig.subplots_adjust(left=0.1, bottom=0.1, right=1.5, top=1.1, hspace=0.3, wspace=0.4)
    
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    
    # vel_th = v*1e-3*np.array([1, 1])
    # rad_th = R*1e-3*np.array([1, 1])
    
    # ax1.plot(vel, autocorr[:,jmax], "b")
    # ax1.plot(vel_th, autocorr[imax,jmax]*np.array([-2, 2]), "--", color="grey")
    # ax1.set_ylim(autocorr[imax,jmax]*np.array([-0.5, 1.5]))
    
    # ax1.set_xlabel(r"Velocity [km]")
    # ax1.set_ylabel(r"$MF/ \sigma$")
    
    # ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    # ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # ax1.xaxis.set_minor_locator(AutoMinorLocator())
    
    
    # ax2.plot(rad_th, autocorr[imax,jmax]*np.array([-2, 2]), "--", color="grey")
    # ax2.plot(radii, autocorr[imax,:], "b")
    # ax2.set_ylim(autocorr[imax,jmax]*np.array([-0.5, 1.5]))

    # ax2.set_xlabel(r"Range [km/s]")
    # ax2.set_ylabel(r"$MF/ \sigma$")
    
    # ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    # ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # ax2.xaxis.set_minor_locator(AutoMinorLocator())

def get_object_passes(t_slice, end_t, p, logger, max_points=100):
    # RADAR definition
    eiscat3d = radars.eiscat3d
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
    orbits_a = np.array([7200, 8500, 12000, 10000])*1e3 # km
    orbits_i = np.array([80, 105, 105, 80]) # deg
    orbits_raan = np.array([86, 160, 180, 90]) # deg
    orbits_aop = np.array([0, 50, 40, 55]) # deg
    orbits_mu0 = np.array([60, 5, 30, 8]) # deg
    obj_id = 0

    p.start('Total')
    p.start('object_initialization')

    # Object instanciation
    logger.info("test_tracker_controller -> creating new object\n")

    target = space_object.SpaceObject(
            Prop_cls,
            propagator_options = Prop_opts,
            a = orbits_a[obj_id], 
            e = 0.1,
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
    t_states = equidistant_sampling(
        orbit = target.state, 
        start_t = 0, 
        end_t = end_t, 
        max_dpos=100e3,
    )
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

    return t_states, object_states, eiscat_passes


if __name__ == "__main__":
    main()
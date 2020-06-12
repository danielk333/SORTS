#!/usr/bin/env python
#
#
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('.'))

import numpy.testing as nt
import numpy as n

def test_envisat_detection():
    from mpi4py import MPI
    
    # SORTS imports CORE
    import population_library as plib
    from simulation import Simulation

    #SORTS Libraries
    import radar_library as rlib
    import radar_scan_library as rslib
    import scheduler_library as schlib
    import antenna_library as alib
    import rewardf_library as rflib

    #SORTS functions
    import ccsds_write
    import dpt_tools as dpt

    sim_root = './tests/tmp_test_data/envisat_sim_test'

    radar = rlib.eiscat_uhf()
    radar.set_FOV(max_on_axis=30.0, horizon_elevation=25.0)

    scan = rslib.beampark_model(
        lat = radar._tx[0].lat,
        lon = radar._tx[0].lon,
        alt = radar._tx[0].alt, 
        az = 90.0,
        el = 75.0,
    )
    radar.set_scan(scan)

    #tle files for envisat in 2016-09-05 to 2016-09-07 from space-track.
    TLEs = [
        ('1 27386U 02009A   16249.14961597  .00000004  00000-0  15306-4 0  9994',
        '2 27386  98.2759 299.6736 0001263  83.7600 276.3746 14.37874511760117'),
        ('1 27386U 02009A   16249.42796553  .00000002  00000-0  14411-4 0  9997',
        '2 27386  98.2759 299.9417 0001256  82.8173 277.3156 14.37874515760157'),
        ('1 27386U 02009A   16249.77590267  .00000010  00000-0  17337-4 0  9998',
        '2 27386  98.2757 300.2769 0001253  82.2763 277.8558 14.37874611760201'),
        ('1 27386U 02009A   16250.12384028  .00000006  00000-0  15974-4 0  9995',
        '2 27386  98.2755 300.6121 0001252  82.5872 277.5467 14.37874615760253'),
        ('1 27386U 02009A   16250.75012691  .00000017  00000-0  19645-4 0  9999',
        '2 27386  98.2753 301.2152 0001254  82.1013 278.0311 14.37874790760345'),
    ]

    pop = plib.tle_snapshot(TLEs, sgp4_propagation=True)

    pop['d'] =  n.sqrt(4*2.3*4/n.pi)
    pop['m'] = 2300.
    pop['C_R'] = 1.0
    pop['C_D'] = 2.3
    pop['A'] = 4*2.3

    ccsds_file = './data/uhf_test_data/events/2002-009A-2016-09-06_08:27:08.tdm'

    obs_data = ccsds_write.read_ccsds(ccsds_file)
    jd_obs = dpt.mjd_to_jd(dpt.npdt2mjd(obs_data['date']))

    jd_sort = jd_obs.argsort()
    jd_obs = jd_obs[jd_sort]

    jd_det = jd_obs[0]

    pop.delete([0,1,2,4]) #now just best ID left

    jd_pop = dpt.mjd_to_jd(pop['mjd0'][0])
    tt_obs = (jd_obs - jd_pop)*3600.0*24.0

    sim = Simulation(
        radar = radar,
        population = pop,
        root = sim_root,
        scheduler = schlib.dynamic_scheduler,
    )

    sim.observation_parameters(
        duty_cycle=0.125,
        SST_fraction=1.0,
        tracking_fraction=0.0,
        SST_time_slice=0.2,
    )

    sim.run_observation(jd_obs[-1] - jd_pop + 1.0)

    sim.print_maintenance()
    sim.print_detections()


    sim.set_scheduler_args(
        logger = sim.logger,
    )

    sim.run_scheduler()
    sim.print_tracks()
    
    print(sim.catalogue.tracklets[0]['t'])
    print(jd_obs)

    shutil.rmtree(sim_root)
    assert False


if __name__=='__main__':

    test_envisat_detection()
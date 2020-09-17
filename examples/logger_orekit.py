#!/usr/bin/env python

'''
Profiling components
======================

'''
import numpy as np

import sorts

logger = sorts.profiling.get_logger('orekit')

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'

def run_prop():
    prop = sorts.propagator.Orekit(
        orekit_data = orekit_data, 
        settings=dict(
            in_frame='Orekit-ITRF',
            out_frame='Orekit-EME',
            drag_force = False,
            radiation_pressure = False,
        ),
        logger = logger,
    )

    state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
    t = np.linspace(0,3600*24.0*2,num=5000)
    mjd0 = 53005

    states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)
    return states

states = run_prop()

print('\n SETTING DEBUG MODE \n')
sorts.profiling.term_level(logger, 'DEBUG')

states = run_prop()

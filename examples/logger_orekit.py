#!/usr/bin/env python

'''
====================
Profiling components
====================

This example showcases the use of the sorts.logging module to log the computation status of the ``orekit``
propagator in real time. 
'''
import pathlib
import numpy as np

import sorts

# initialization of the logger
logger = sorts.common.profiling.get_logger('orekit')

# gets the orekit configuration
try:
    pth = pathlib.Path(__file__).parent.resolve()
except NameError:
    pth = pathlib.Path('.').parent.resolve()
pth = pth / 'data' / 'orekit-data-master.zip'
if not pth.is_file():
    sorts.propagator.Orekit.download_quickstart_data(pth, verbose=True)

def run_prop():
    ''' Starts the propagation of the Orekit propagator with the logging option enabled. '''
    prop = sorts.targets.propagator.Orekit(
        orekit_data = pth, 
        settings=dict(
            in_frame='Orekit-ITRF',
            out_frame='Orekit-EME',
            drag_force = False,
            radiation_pressure = False,
        ),
        logger = logger,
    )

    # initial conditions
    state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
    t = np.linspace(0,3600*24.0*2,num=5000)
    mjd0 = 53005

    # propagate states
    states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)
    return states

# run propagator with default logging level enabled
states = run_prop()

print('\n SETTING DEBUG MODE \n')

# run propagator with debug level enabled
sorts.profiling.term_level(logger, 'DEBUG')
states = run_prop()

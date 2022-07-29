#!/usr/bin/env python

'''
============================================
Example of the Simulation command line usage
============================================

This example showcases the basic ``sorts.simulation`` commands used in the simulation helper.

The simulation implemented in this example possesses two distinct steps :
    - step1 : prints "This is step 1!"
    - plot : prints "Now we are doing "plotting" with dark_style=True/False!" 

.. note::
    All the command lines in this example are executed from the ``SORTS/`` directory.

1. to run the first step, simply run :

>>> python examples/simulation_command_line.py run step1
2022-07-26 12:36:21.883 ALWAYS  ; Simulation:parse_cmd:parsing command
2022-07-26 12:36:21.887 INFO    ; Simulation:run:step1

This is step 1!

2022-07-26 12:36:21.887 INFO    ; Simulation:run:step1 [completed]

2. to run the plot step, run :
>>> python examples/simulation_command_line.py run step1
2022-07-26 12:36:21.883 ALWAYS  ; Simulation:parse_cmd:parsing command
2022-07-26 12:36:21.887 INFO    ; Simulation:run:step1

Now we are doing "plotting" with dark_style=False!


2022-07-26 12:36:21.887 INFO    ; Simulation:run:step1 [completed]

.. note::
    In this example the custom --darkstyle option (implemented in the :class:`MySim` class) is disabled, to enable 
    this option, simply call :

    >>> python examples/simulation_command_line.py run plot --dark-style
    2022-07-26 12:44:40.088 ALWAYS  ; Simulation:parse_cmd:parsing command
    2022-07-26 12:44:40.092 INFO    ; Simulation:run:plot

    Now we are doing "plotting" with dark_style=True!

    2022-07-26 12:44:40.092 INFO    ; Simulation:run:plot [completed]

To access the help option of the simulation feature, run : 

>>> python examples/simulation_command_line.py --help
2022-07-26 12:35:49.803 ALWAYS  ; Simulation:parse_cmd:parsing command
usage: simulation_command_line.py [-h] {run,cmd} ...

Simulation command-line interface

positional arguments:
  {run,cmd}   Actions
    run       Run simulation
    cmd       Simulation command

optional arguments:
  -h, --help  show this help message and exit
'''
import pathlib
import sorts

class MySim(sorts.Simulation):
    ''' Custom simulation class.

    Implements a custom simulation class to showcase the basic use of the simulation helper module
    of ``sorts``.
    '''
    def __init__(self, *args, **kwargs):
        ''' Class constructor. '''
        super().__init__(*args, **kwargs)

        # add the steps to the simulation
        self.steps['step1'] = self.step1
        self.steps['plot'] = self.plot

        # add a command-line arguments using the exact syntax of argparse
        # these are all sent to the `**kw` of the step(s) being executed
        # to enable --dark-style, add --dark-style to the end of the command
        self.add_cmd_argument('--dark-style', action='store_true')


    def step1(self, **kw):
        ''' First step of the simulation. '''
        print('\nThis is step 1!\n')


    def plot(self, **kw):
        ''' Second step of the simulation. '''
        dark_style = kw.get('dark_style', False)

        print(f'\nNow we are doing "plotting" with dark_style={dark_style}!\n')


# if this file is run as a script
if __name__=='__main__':
    # initialize the simulation
    sim = MySim(
        scheduler = None, 
        root = None, # this disables all disk persistency
    )

    # and parse the input args and execute like a command-line interface
    sim.parse_cmd()
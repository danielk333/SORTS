#!/usr/bin/env python

'''
Example of the Simulation command line usage
================================================
'''
import pathlib
import sorts

class MySim(sorts.Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.steps['step1'] = self.step1
        self.steps['plot'] = self.plot

        #add a command-line arguments using the exact syntax of argparse
        #these are all sent to the `**kw` of the step(s) being executed
        self.add_cmd_argument('--dark-style', action='store_true')


    def step1(self, **kw):
        print('\nThis is step 1!\n')


    def plot(self, **kw):
        dark_style = kw.get('dark_style', False)

        print(f'\nNow we are doing "plotting" with dark_style={dark_style}!\n')


#if thiis file is run as a script
if __name__=='__main__':
    
    #initialize the simulation
    sim = MySim(
        scheduler = None, 
        root = None, #this disables all disk persistency
    )

    #and parse the input args and execute like a command-line interface
    sim.parse_cmd()
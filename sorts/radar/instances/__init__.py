#!/usr/bin/env python

'''
'''

from .eiscat_3d import gen_eiscat3d

radar_instances = ['eiscat3d', 'eiscat3d_interp']

def __getattr__(name):
    if name == 'eiscat3d':
        return gen_eiscat3d(beam='array')
    elif name == 'eiscat3d_interp':
        return gen_eiscat3d(beam='interp')
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

#!/usr/bin/env python

'''
'''

from .eiscat_3d import gen_eiscat3d
from .tsdr import gen_tromso_space_debris_radar

radar_instances = ['eiscat3d', 'eiscat3d_interp']

class RadarSystemsGetter:
    '''

    :eiscat3d:

    TODO: Docstring

    :eiscat3d_interp:

    TODO: Docstring

    '''
    instances = radar_instances
    __all__ = radar_instances

    def __getattr__(self, name):
        if name == 'eiscat3d':
            return gen_eiscat3d(beam='array')
        elif name == 'eiscat3d_interp':
            return gen_eiscat3d(beam='interp')
        elif name == 'tsdr':
            return gen_tromso_space_debris_radar(fence=False)
        elif name == 'tsdr_fence':
            return gen_tromso_space_debris_radar(fence=True)
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

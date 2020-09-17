#!/usr/bin/env python

'''
'''

from .eiscat_3d import gen_eiscat3d, gen_eiscat3d_demonstrator
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
        elif name == 'eiscat3d_demonstrator':
            return gen_eiscat3d_demonstrator(beam='array')
        elif name == 'eiscat3d_demonstrator_interp':
            return gen_eiscat3d_demonstrator(beam='interp')
        elif name == 'tsdr':
            return gen_tromso_space_debris_radar(fence=False, phased=False)
        elif name == 'tsdr_fence':
            return gen_tromso_space_debris_radar(fence=True, phased=False)
        elif name == 'tsdr_phased':
            return gen_tromso_space_debris_radar(fence=False, phased=True)
        elif name == 'tsdr_phased_fence':
            return gen_tromso_space_debris_radar(fence=True, phased=True)
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

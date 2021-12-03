#!/usr/bin/env python

'''Defines all the pre-configured radar instances available and provides a simple "getter" instance.


    :eiscat3d:

    TODO: Docstring

    :eiscat3d_interp:

    TODO: Docstring

'''

from .eiscat_3d import gen_eiscat3d, gen_eiscat3d_demonstrator
from .tsdr import gen_tromso_space_debris_radar
from .eiscat_uhf import gen_eiscat_uhf
from .eiscat_esr import gen_eiscat_esr
from .mock import gen_mock

radar_instances = [
    'eiscat3d', 
    'eiscat3d_interp',
    'mock',
    'eiscat3d_demonstrator'
    'eiscat3d_demonstrator_interp',
    'tsdr',
    'tsdr_fence',
    'tsdr_phased',
    'tsdr_phased_fence',
    'eiscat_uhf',
    'eiscat_esr',
]


class RadarSystemsGetter:
    '''
    '''
    instances = radar_instances
    __all__ = radar_instances

    def __getattr__(self, name):
        if name == 'eiscat3d':
            return gen_eiscat3d(beam='array')
        if name == 'mock':
            return gen_mock()
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
        elif name == 'eiscat_uhf':
            return gen_eiscat_uhf()
        elif name == 'eiscat_esr':
            return gen_eiscat_esr()
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'\n(valid names: {radar_instances}")

#!/usr/bin/env python

'''Defines function for reading the MASTER catalog files.

'''

import numpy as np

from .population import Population



def master_catalog(
        input_file,
        mjd0 = 54952.0,
        sort=True,
        propagator = None,
        propagator_options = {},
        propagator_args = {},
    ):
    '''Return the master catalog specified in the input file as a population instance. The catalog only contains the master sampling objects and not an actual realization of the population using the factor.

    The format of the input master files is:

        0. ID
        1. Factor
        2. Mass [kg]
        3. Diameter [m]
        4. m/A [kg/m2]
        5. a [km]
        6. e
        7. i [deg]
        8. RAAN [deg]
        9. AoP [deg]
        10. M [deg]


    :param str input_file: Path to the input MASTER file.
    :param bool sort: If :code:`True` sort according to diameters in descending order.
    :param float mjd0: The epoch of the catalog file in Modified Julian Days.
    :param PropagatorBase propagator: Propagator class pointer used for :class:`space_object.SpaceObject`.
    :param dict propagator_options: Propagator initialization keyword arguments.
    

    :return: Master catalog
    :rtype: population.Population
    '''
    master_raw = np.genfromtxt(input_file);
    i = [0,5,6,7,8,9,10]
    var = ['oid', 'a', 'e', 'i', 'raan', 'aop', 'mu0']

    master = Population(
        fields = ['oid', 'a', 'e', 'i', 'aop', 'raan', 'mu0', 'mjd0', 'A', 'm', 'd', 'C_D', 'C_R', 'Factor', 'MASTER-ID'],
        dtypes = ['int'] + ['float64']*13 + ['int'],
        space_object_fields = ['A', 'm', 'd', 'C_D', 'C_R'],
        state_fields = ['a', 'e', 'i', 'aop', 'raan', 'mu0'],
        epoch_field = {'field': 'mjd0', 'format': 'mjd', 'scale': 'utc'},
        propagator = propagator,
        propagator_options = propagator_options,
        propagator_args = propagator_args,
    )

    master.allocate(master_raw.shape[0])

    for ind, key in zip(i, var):
        master.data[key] = master_raw[:, ind]

    master.data['a'] *= 1e3 #km to m
    master.data['mjd0'] = mjd0
    master.data['A'] = np.divide(master_raw[:, 2], master_raw[:, 4])
    master.data['m'] = master_raw[:, 2]
    master.data['d'] = master_raw[:, 3]
    master.data['C_D'] = 2.3
    master.data['C_R'] = 1.0
    master.data['Factor'] = master_raw[:, 1]
    master.data['MASTER-ID'] = master_raw[:, 0]

    diams = master_raw[:, 3]

    if sort:
        idxs = np.argsort(diams)[::-1]
    else:
        idxs = np.arange(len(diams))

    master.data = master.data[idxs]
    
    return master



def master_catalog_factor(
        master_base,
        copy = True,
        treshhold = 0.01,
        seed=None,
    ):
    '''Returns a random realization of the master population specified by the input file/population. In other words, each sampling object in the catalog is sampled a "factor" number of times with random mean anomalies to create the population.

    :param population.Population master_base: A master catalog consisting only of sampling objects. This catalog will be modified and the pointer to it returned.
    :param float treshhold: Diameter limit in meters below witch sampling objects are not included. Can be :code:`None` to skip filtering.
    :param int seed: Random number generator seed given to :code:`numpy.random.seed` to allow for consisted generation of a random realization of the population. If seed is :code:`None` a random seed from high-entropy data is used.
    :param bool copy: Modify the given `master_base` instance or return a modified copy.
    

    :return: Master population
    :rtype: population.Population
    '''
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed=seed)
    
    if copy:
        master = master_base.copy()
    else:
        master = master_base

    if treshhold is not None:
        master.filter('d', lambda d: d >= treshhold)

    full_objs = np.zeros((int(np.sum(np.round(master['Factor']))),master.shape[1]), dtype=np.float)
    
    max_oid = np.max(master['oid'])
    max_factor = np.floor(np.log10(np.max(master['Factor'])))
    oid_extend = 10**(int(np.log10(max_oid))+max_factor+1.0)
    oid_mag = 10**(int(np.log10(max_oid)))

    i=0
    for row in master.data:
        f_int = int(np.round(row[13]))
        if f_int >= 1:
            ip = i+f_int
            for coli, head in enumerate(master.fields):
                full_objs[i:ip,coli] = row[head]

            full_objs[i:ip,0] = np.array(range(i,ip), dtype=np.float)
            full_objs[i:ip,6] = np.random.rand(f_int)*360.0
            i=ip

    master.allocate(full_objs.shape[0])

    master[:,:] = full_objs

    if seed is not None:
        np.random.seed(None)
        np.random.set_state(st0)

    return master

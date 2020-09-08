#!/usr/bin/env python

'''

'''

from .. import frames

def tle_catalog(
        tles,
        keep_tle = False,
        propagator = None,
        propagator_options = {},
        propagator_args = {},
    ):
    '''Reads a TLE-snapshot file and converts the TLE's to orbits in a TEME frame and creates a population file. The BSTAR parameter is saved in column BSTAR (or :code:`_objs[:,12`). A snapshot generally contains several TLE's for the same object thus will this population also contain duplicate objects.
    
    *Numerical propagator assumptions:*
    To propagate with a numerical propagator one needs to make assumptions.
       * Density is :math:`5\cdot 10^3 \;\frac{kg}{m^3}`.
       * Object is a sphere
       * Drag coefficient is 2.3.


    :param str/list tles: Path to the input TLE snapshot file. Or the TLE-set can be given directly as a list of two lines that can be unpacked in a loop, e.g. :code:`[(tle1_l1, tle1_l2), (tle2_l1, tle2_l2)]`.
    :param bool tle_to_state: If :code:`True` 

    :return: TLE snapshot as a Population
    :rtype: sorts.Population
    '''
    if isinstance(tles, str):
        tle_raw = [line.rstrip('\n') for line in open(tles)]
        if len(tle_raw) % 2 != 0:
            raise Exception('Not even number of lines [not TLE compatible]')

        tles = zip(tle_raw[0::2], tle_raw[1::2])


    #FIX THIS
    if tle_to_state:
        pop = Population(
            extra_columns = ['A', 'm', 'd', 'C_D', 'C_R'] + ['line1', 'line2'],
            dtypes = ['float64']*5 + ['U70', 'U70'],
            space_object_uses = [True, True, True, True, True] + [True, True],
        )
    else:
        pop = Population(
            extra_columns = ['A', 'm', 'd', 'C_D', 'C_R'],
            space_object_uses = [True, True, True, True, True],
            propagator = propagator,
            propagator_options = propagator_options,
        )

    pop.allocate(len(tles))


    #FIX THIS TOOO
    for line_id, lines in enumerate(tles):
        line1, line2 = lines

        sat_id = tle.tle_id(line1)
        jd0 = tle.tle_jd(line1)
        mjd0 = dpt.jd_to_mjd(jd0)

        state_TEME, epoch = frames.TLE_to_TEME(line1,line2)
        kep = dpt.cart2kep(state_TEME, m=0.0, M_cent=M_earth, radians=False)
        pop.objs[line_id][1] = kep[0]*1e-3
        pop.objs[line_id][2] = kep[1]
        pop.objs[line_id][3] = kep[2]
        pop.objs[line_id][4] = kep[4]
        pop.objs[line_id][5] = kep[3]
        pop.objs[line_id][6] = dpt.true2mean(kep[5], kep[1], radians=False)
        
        pop.objs[line_id][0] = float(sat_id)
        pop.objs[line_id][7] = mjd0

    if sgp4_propagation:
        for line_id, lines in enumerate(tles):
            line1, line2 = lines
            pop.objs[line_id][13] = line1[:-1]
            pop.objs[line_id][14] = line2[:-1]
    
    for line_id, lines in enumerate(tles):
        line1, line2 = lines

        bstar = tle.tle_bstar(line1)/(propagator_sgp4.SGP4.R_EARTH*1000.0)
        B = bstar*2.0/propagator_sgp4.SGP4.RHO0
        if B < 1e-9:
            rho = 500.0
            C_D = 0.0
            r = 0.1
            A = np.pi*r**2
            m = rho*4.0/3.0*np.pi*r**3
        else:
            C_D = 2.3
            rho = 5.0
            r = (3.0*C_D)/(B*rho)
            A = np.pi*r**2
            m = rho*4.0/3.0*np.pi*r**3

        pop.objs[line_id][8] = A
        pop.objs[line_id][9] = m
        pop.objs[line_id][10] = r*2.0
        pop.objs[line_id][11] = C_D
        pop.objs[line_id][12] = 1.0

    return pop


import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import time

import unittest
import numpy as n
import numpy.testing as nt
import scipy

import population_library as plib
from population import Population
from space_object import SpaceObject
import TLE_tools as tle

class TestPopulationLibrary(unittest.TestCase):

    def test_master(self):
        pop = plib.master_catalog()
        assert len(pop) > 0
        assert isinstance(pop, Population)
        so = pop.get_object(0)
        assert isinstance(so, SpaceObject)
        states = so.get_state(0)
        assert states.shape == (6,1)

        self.assertListEqual( pop.header, ['oid','a','e','i','raan','aop','mu0','mjd0'] + ['A', 'm', 'd', 'C_D', 'C_R', 'Factor'] )

    def test_master_factor(self):
        pop = plib.master_catalog_factor(treshhold = 0.01)
        assert len(pop) > 0
        assert isinstance(pop, Population)
        so = pop.get_object(0)
        states = so.get_state(0)
        assert states.shape == (6,1)
        d_vec = pop['d']
        nt.assert_array_less(n.full( (len(pop),), 0.009, dtype=d_vec.dtype), d_vec)

        self.assertListEqual( pop.header, ['oid','a','e','i','raan','aop','mu0','mjd0'] + ['A', 'm', 'd', 'C_D', 'C_R', 'Factor'] )

    def test_master_factor_cnt(self):
        popf = plib.master_catalog_factor(treshhold = 0.01)
        pop = plib.master_catalog()
        pop.filter('d', lambda x: x >= 0.01)
        self.assertEqual(len(popf), int(n.sum(n.round(pop['Factor']))))


    def test_tle_snapshot(self):
        
        line1 = '1     5U 58002B   18002.33074547 +.00000199 +00000-0 +22793-3 0  9999'
        line2 = '2     5 034.2599 111.6722 1847887 275.6635 063.7215 10.84778261107822'

        pop = plib.tle_snapshot('data/uhf_test_data/tle-201801.txt', sgp4_propagation=True)

        #find row number of above sat
        sat_id = tle.tle_id(line1)
        oids = pop['oid']
        ind = n.argmin(n.abs(oids - float(sat_id)))

        self.assertEqual(pop.header[-1], 'line2')
        self.assertEqual(pop.header[-2], 'line1')

        row0 = pop[ind]
        self.assertEqual(str(row0['line1']), line1)
        self.assertEqual(str(row0['line2']), line2)

        obj0 = pop.get_object(ind)
        self.assertEqual(int(obj0.oid), int(sat_id))

        ecef = obj0.get_state(0)
        self.assertEqual(ecef.shape[0], 6)

        popp = plib.tle_snapshot('data/uhf_test_data/tle-201801.txt', sgp4_propagation=False)

        self.assertEqual(popp.header[8:], ['A', 'm', 'd', 'C_D', 'C_R'])

        obj0 = popp.get_object(ind)
        self.assertEqual(int(obj0.oid), int(sat_id))

        ecef = obj0.get_state(0)
        self.assertEqual(ecef.shape[0], 6)

        self.assertEqual(len(pop),len(popp))


if __name__ == '__main__':
    unittest.main(verbosity=2)
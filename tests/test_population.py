import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as np
import numpy.testing as nt

import types

from sorts.targets import Population
from sorts.targets import SpaceObject

class TestPopulation(unittest.TestCase):

    def test_constructor(self):
        pop = Population()
        self.assertEqual(len(pop.fields), len(Population._default_fields))
        self.assertEqual(pop.data.shape, (0,))
        self.assertEqual(len(pop.data.dtype), len(Population._default_fields))
        for ind in range(len(pop.data.dtype)):
            self.assertEqual(pop.data.dtype[ind], np.dtype(Population._default_dtype))

    def test_constructor_arguemnts(self):
        pop = Population(
            fields = ['x', 'y'],
            dtypes = ['float64', 'float32'],
        )
        self.assertEqual(len(pop.fields), 2)
        self.assertEqual(pop.data.shape, (0,))
        self.assertEqual(len(pop.data.dtype), 2)

        self.assertEqual(pop.data.dtype[0], np.dtype('float64'))
        self.assertEqual(pop.data.dtype[1], np.dtype('float32'))

    def test_allocate(self):
        pop = Population()
        pop.allocate(10)
        self.assertEqual(pop.data.shape, (10,))
        self.assertEqual(len(pop.data.dtype), len(Population._default_fields))
        for ind in range(len(pop.data.dtype)):
            self.assertEqual(pop.data.dtype[ind], np.dtype(Population._default_dtype))
        self.assertEqual(len(pop), 10)

        pop = Population(
            fields = ['x', 'y'],
            dtypes = ['float64', 'float32'],
        )

        pop.allocate(10)
        self.assertEqual(len(pop.fields), 2)
        self.assertEqual(pop.data.shape, (10,))
        self.assertEqual(len(pop.data.dtype), 2)
        self.assertEqual(pop.data.dtype[0], np.dtype('float64'))
        self.assertEqual(pop.data.dtype[1], np.dtype('float32'))
        self.assertEqual(len(pop), 10)

    def test_shape(self):

        pop = Population()
        shape = pop.shape

        self.assertEqual(shape, (0,len(Population._default_fields)))
        pop.allocate(123)
        shape = pop.shape        

        self.assertEqual(shape, (123,len(Population._default_fields)))


        pop = Population(
            fields = ['x', 'y'],
            dtypes = ['float64', 'float32'],
        )
        shape = pop.shape

        self.assertEqual(shape, (0,2))
        pop.allocate(123)
        shape = pop.shape

        self.assertEqual(shape, (123,2))

    def test_column_order(self):

        pop = Population()

        ref = Population._default_fields
        keys = pop.data.dtype.fields.keys()

        for ind in range(len(pop.fields)):
            assert pop.fields[ind] in keys
            assert pop.fields[ind] in ref

        pop.allocate(1)
        pop[0,0] = 1.0
        self.assertEqual(pop['oid'][0], 1.0)
        vec = np.arange(len(pop.fields))

        pop[0,:] = vec.copy()
        for ind in range(len(pop.fields)):
            nt.assert_almost_equal(pop[0,ind], vec[ind])
            nt.assert_almost_equal(pop[ref[ind]][0], vec[ind])
            nt.assert_almost_equal(pop[0,ind], pop[ref[ind]][0])

        pop = Population(
            fields = ['x', 'y'],
            dtypes = ['float64', 'float32'],
        )

        ref = ['x', 'y']
        keys = pop.data.dtype.fields.keys()
        
        for ind in range(len(pop.fields)):
            assert pop.fields[ind] in keys
            assert pop.fields[ind] in ref

        pop.allocate(1)
        pop[0,0] = 1.0
        self.assertEqual(pop['x'][0], 1.0)
        vec = np.arange(len(pop.fields))
        
        pop[0,:] = vec.copy()
        for ind in range(len(pop.fields)):
            nt.assert_almost_equal(pop[0,ind], vec[ind])
            nt.assert_almost_equal(pop[ref[ind]][0], vec[ind])
            nt.assert_almost_equal(pop[0,ind], pop[ref[ind]][0])


    def test_iter(self):
        pop = Population()
        pop.allocate(10)

        pop['oid'] = np.arange(10)

        itres = 0
        for ind, row in enumerate(pop.data):
            self.assertEqual(int(row[0]), ind)
            self.assertEqual(len(row), 8)
            itres += 1
        self.assertEqual(itres, 10)


    def test_space_object(self):
        from sorts.targets.propagator import SGP4

        pop = Population(propagator = SGP4)
        pop.allocate(1)
        for ind in range(1):
            pop.data[ind] = (ind,
                7000e3,
                0.0,
                69,
                ind,
                0.0,
                0.0,
                57125.7729,
            )

        pop.in_frame = 'TEME'
        pop.out_frame = 'TEME'

        obj = pop.get_object(0)

        assert obj.in_frame == 'TEME'
        assert obj.out_frame == 'TEME'

        assert isinstance(obj, SpaceObject)

        state = obj.get_state(0)
        assert state.shape == (6,1)

        #3 decimals to account for SGP4 bad conversion
        nt.assert_almost_equal(state, obj.orbit.cartesian, decimal=3) 

        ref = np.array([
            0,
            obj.orbit.a,
            obj.orbit.e,
            obj.orbit.i,
            obj.orbit.omega,
            obj.orbit.Omega,
            obj.orbit.anom,
            obj.epoch.mjd,
        ], dtype=Population._default_dtype)

        for ind in range(8):
            nt.assert_almost_equal(pop.data[0][ind], ref[ind], decimal=6)


    def test_filter(self):
        pop = Population()
        pop.allocate(10)
        for ind in range(10):
            pop.data[ind] = (ind,
                7000e3,
                0.0,
                69,
                ind,
                0.0,
                0.0,
                57125.7729,
            )

        pop.filter('raan', lambda theta: theta > 5.5)
        assert len(pop) == 4


    def test_generator(self):
        class MockProp:
            pass

        pop = Population(propagator = MockProp)
        pop.allocate(10)
        for ind in range(10):
            pop.data[ind] = (ind,
                7000,
                0.0,
                69,
                ind,
                0.0,
                0.0,
                57125.7729,
            )
        gen = pop.generator

        assert isinstance(gen, types.GeneratorType)

        lst = [obj for obj in gen]
        assert len(lst) == 10
        for ind, obj in enumerate(lst):
            assert isinstance(obj, SpaceObject)
            nt.assert_almost_equal(obj.oid, float(ind), decimal = 7)


    def test_set_item(self):
        pop0 = Population()

        pop = Population(
            fields = Population._default_fields + ['m', 'color'],
            dtypes = [Population._default_dtype]*len(Population._default_fields) + ['float64', 'U3'],
        )

        pop0.allocate(2)
        pop.allocate(2)

        pop[1,5] = 1
        nt.assert_almost_equal(pop.data[1][pop.fields[5]], 1)

        pop[1,9] = 'WAT'
        pop0[0] = np.random.rand(len(pop.fields))
        pop['m'] = np.random.randn(len(pop))
        pop[:,3] = np.ones((len(pop),1))*3
        pop[:,5:8] = np.ones((len(pop),3))*4

        self.assertEqual(str(pop.data[1]['color']), "WAT")
        self.assertEqual(str(pop.data[0]['color']), '')
        nt.assert_almost_equal(pop.data[1][pop.fields[5]], 4)
        nt.assert_almost_equal(pop.data[1][pop.fields[3]], 3)

    def test_get_item_nan(self):

        pop = Population(
            fields = Population._default_fields + ['m', 'color'],
            dtypes = [Population._default_dtype]*len(Population._default_fields) + ['float64', 'U20'],
        )

        pop.allocate(2)

        test = pop[0,:]
        assert np.isnan(test[9])

    def test_get_item(self):

        pop = Population(
            fields = Population._default_fields + ['m', 'color'],
        )

        pop.allocate(10)
        for ind in range(10):
            pop.data[ind] = (ind,
                7000,
                0.0,
                69,
                ind,
                0.0,
                0.0,
                57125.7729,
                123,
                1223,
            )

        ms = pop['m']
        es = pop['e']

        row = pop[0]

        assert len(ms) == 10
        assert len(row) == len(pop.fields)

        for ind in range(len(pop.fields)):
            nt.assert_almost_equal(row[ind], pop.data[0][ind], decimal=9)
        for ind in range(len(pop)):
            nt.assert_almost_equal(es[ind], pop.data['e'][ind], decimal=9)
        for ind in range(len(pop)):
            nt.assert_almost_equal(ms[ind], pop.data[ind][8], decimal=9)

        vec = pop[4,:-1]
        assert len(vec) == len(pop.fields)-1
        vec = pop[:,3]
        assert len(vec) == len(pop)
        point = pop[3,3]
        mat = pop[:,:]
        assert mat.shape == pop.shape
        mat = pop[2:6,1:2:-1]



if __name__ == '__main__':
    unittest.main(verbosity=2)
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy.testing as nt

import types

from population import Population
from space_object import SpaceObject

class TestPopulation(unittest.TestCase):

    def test_constructor(self):
        pop = Population()
        self.assertEqual(len(pop.header), 8)
        self.assertEqual(pop.objs.shape, (0,))
        self.assertEqual(len(pop.objs.dtype), 8)
        for ind in range(len(pop.objs.dtype)):
            self.assertEqual(pop.objs.dtype[ind], n.dtype(Population._default_dtype))

    def test_constructor_arguemnts(self):
        pop = Population(
            name='two objects',
            extra_columns = ['m', 'color'],
            dtypes = ['f', 'U10'],
            space_object_uses = [True, False],
        )

    def test_allocate(self):
        pop = Population()
        pop.allocate(10)
        self.assertEqual(pop.objs.shape, (10,))
        self.assertEqual(len(pop.objs.dtype), 8)
        for ind in range(len(pop.objs.dtype)):
            self.assertEqual(pop.objs.dtype[ind], n.dtype(Population._default_dtype))
        self.assertEqual(len(pop), 10)

        pop = Population(
            name='two objects',
            extra_columns = ['m', 'color'],
            dtypes = ['f', 'U10'],
            space_object_uses = [True, False],
        )

        pop.allocate(10)
        self.assertEqual(len(pop.header), 10)
        self.assertEqual(pop.objs.shape, (10,))
        self.assertEqual(len(pop.objs.dtype), 10)
        for ind in range(8):
            self.assertEqual(pop.objs.dtype[ind], n.dtype(Population._default_dtype))
        self.assertEqual(pop.objs.dtype[8], n.dtype('f'))
        self.assertEqual(pop.objs.dtype[9], n.dtype('U10'))
        self.assertEqual(len(pop), 10)

    def test_shape(self):

        pop = Population()
        shape = pop.shape

        self.assertEqual(shape, (0,8))
        pop.allocate(123)
        shape = pop.shape        

        self.assertEqual(shape, (123,8))

        pop = Population(
            name='two objects',
            extra_columns = ['m', 'color'],
            dtypes = ['f', 'U10'],
            space_object_uses = [True, False],
        )
        shape = pop.shape

        self.assertEqual(shape, (0,10))
        pop.allocate(123)
        shape = pop.shape

        self.assertEqual(shape, (123,10))

    def test_column_order(self):

        pop = Population()

        ref = ['oid','a','e','i','raan','aop','mu0','mjd0']
        keys = pop.objs.dtype.fields.keys()

        for ind in range(len(pop.header)):
            assert ref[ind] in keys
            assert ref[ind] in pop.header

        pop.allocate(1)
        pop[0,0] = 1.0
        self.assertEqual(pop['oid'][0], 1.0)
        vec = n.arange(len(pop.header))
        pop[0,:] = vec.copy()
        for ind in range(len(pop.header)):
            nt.assert_almost_equal(pop[0,ind], vec[ind])
            nt.assert_almost_equal(pop[ref[ind]][0], vec[ind])
            nt.assert_almost_equal(pop[0,ind], pop[ref[ind]][0])

        pop = Population(
            name='two objects',
            extra_columns = ['m', 'color'],
            dtypes = ['f', 'U10'],
            space_object_uses = [True, False],
        )

        ref = ['oid','a','e','i','raan','aop','mu0','mjd0'] + ['m', 'color']
        keys = pop.objs.dtype.fields.keys()
        for ind in range(len(pop.header)):
            assert ref[ind] in keys
            assert ref[ind] in pop.header

        pop.allocate(1)
        pop[0,0] = 1.0
        self.assertEqual(pop['oid'][0], 1.0)
        vec = n.arange(len(pop.header)-1)
        pop[0,:-1] = vec.copy()
        for ind in range(len(pop.header)-1):
            nt.assert_almost_equal(pop[0,ind], vec[ind])
            nt.assert_almost_equal(pop[ref[ind]][0], vec[ind])
            nt.assert_almost_equal(pop[0,ind], pop[ref[ind]][0])

    def test_iter(self):
        pop = Population()
        pop.allocate(10)

        pop['oid'] = n.arange(10)

        itres = 0
        for ind, row in enumerate(pop):
            self.assertEqual(int(row[0]), ind)
            self.assertEqual(len(row), 8)
            itres += 1
        self.assertEqual(itres, 10)

    def test_space_object(self):
        pop = Population()
        pop.allocate(1)
        for ind in range(1):
            pop.objs[ind] = (ind,
                7000,
                0.0,
                69,
                ind,
                0.0,
                0.0,
                57125.7729,
            )

        obj = pop.get_object(0)

        assert isinstance(obj, SpaceObject)

        state = obj.get_state(0)
        assert state.shape == (6,1)

        ref = n.array([
            0,
            obj.a,
            obj.e,
            obj.i,
            obj.raan,
            obj.aop,
            obj.mu0,
            obj.mjd0,
        ], dtype=Population._default_dtype)

        for ind in range(8):
            nt.assert_almost_equal(pop.objs[0][ind], ref[ind], decimal=6)


    def test_filter(self):
        pop = Population()
        pop.allocate(10)
        for ind in range(10):
            pop.objs[ind] = (ind,
                7000,
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
        pop = Population()
        pop.allocate(10)
        for ind in range(10):
            pop.objs[ind] = (ind,
                7000,
                0.0,
                69,
                ind,
                0.0,
                0.0,
                57125.7729,
            )
        gen = pop.object_generator()

        assert isinstance(gen, types.GeneratorType)

        lst = [obj for obj in gen]
        assert len(lst) == 10
        for ind, obj in enumerate(lst):
            assert isinstance(obj, SpaceObject)
            nt.assert_almost_equal(obj.oid, float(ind), decimal = 7)


    def test_set_item(self):
        pop0 = Population(
                name='test objects',
            )

        pop = Population(
                name='two objects',
                extra_columns = ['m', 'color'],
                dtypes = ['float64', 'U20'],
                space_object_uses = [True, False],
            )

        pop0.allocate(2)
        pop.allocate(2)

        pop[1,5] = 1
        nt.assert_almost_equal(pop.objs[1][pop.header[5]], 1)

        pop[1,9] = 'WAT'
        pop0[0] = n.random.rand(len(pop.header))
        pop['m'] = n.random.randn(len(pop))
        pop[:,3] = n.ones((len(pop),1))*3
        pop[:,5:8] = n.ones((len(pop),3))*4

        self.assertEqual(str(pop.objs[1]['color']), 'WAT')
        self.assertEqual(str(pop.objs[0]['color']), '')
        nt.assert_almost_equal(pop.objs[1][pop.header[5]], 4)
        nt.assert_almost_equal(pop.objs[1][pop.header[3]], 3)

    def test_get_item_nan(self):
        pop = Population(
                name='two objects',
                extra_columns = ['m', 'color'],
                dtypes = ['float64', 'U20'],
                space_object_uses = [True, False],
            )
        
        pop.allocate(2)

        test = pop[0,:]
        assert n.isnan(test[9])

    def test_get_item(self):
        pop = Population(
            name='two objects',
            extra_columns = ['m', 'color'],
            space_object_uses = [True, False],
        )
        pop.allocate(10)
        for ind in range(10):
            pop.objs[ind] = (ind,
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
        assert len(row) == len(pop.header)

        for ind in range(len(pop.header)):
            nt.assert_almost_equal(row[ind], pop.objs[0][ind], decimal=9)
        for ind in range(len(pop)):
            nt.assert_almost_equal(es[ind], pop.objs['e'][ind], decimal=9)
        for ind in range(len(pop)):
            nt.assert_almost_equal(ms[ind], pop.objs[ind][8], decimal=9)

        vec = pop[4,:-1]
        assert len(vec) == len(pop.header)-1
        vec = pop[:,3]
        assert len(vec) == len(pop)
        point = pop[3,3]
        mat = pop[:,:]
        assert mat.shape == pop.shape
        mat = pop[2:6,1:2:-1]



if __name__ == '__main__':
    unittest.main(verbosity=2)
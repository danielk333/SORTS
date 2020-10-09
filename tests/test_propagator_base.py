import sys
import os
from copy import copy
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy.testing as nt

from sorts import Propagator



class TestBaseProp(unittest.TestCase):

    def test_base_prop_methods(self):

        class new_propagator(Propagator):
            def propagate(self, t, state0, epoch, **kwargs):
                pass

        prop = new_propagator()
        assert prop.propagate(0,0,0) is None


    def test_meta_raise_no_method(self):

        class no_propagate(Propagator):
            pass

        with self.assertRaises(TypeError):
            prop = no_propagate()

    def test_meta_raise_wrong_method(self):

        class wrong_form(Propagator):
            def propagate(self, state0):
                pass

        with self.assertRaises(AssertionError):
            prop = wrong_form()



    def test_base_frame_property(self):

        class new_propagator(Propagator):

            DEFAULT_SETTINGS = copy(Propagator.DEFAULT_SETTINGS)
            DEFAULT_SETTINGS.update(
                dict(
                    out_frame = 'out',
                    in_frame = 'in',
                )
            )
            def propagate(self, t, state0, epoch, **kwargs):
                pass

        prop = new_propagator()
        assert prop.in_frame == 'in'
        assert prop.out_frame == 'out'

        prop.in_frame = 'in2'
        prop.out_frame = 'out2'

        assert prop.in_frame == 'in2'
        assert prop.out_frame == 'out2'


    def test_base_frame_property_error(self):

        class new_propagator(Propagator):
            def propagate(self, t, state0, epoch, **kwargs):
                pass

        prop = new_propagator()
        with self.assertRaises(AttributeError):
            a = prop.in_frame
        with self.assertRaises(AttributeError):
            a = prop.out_frame
        with self.assertRaises(AttributeError):
            prop.in_frame = ''
        with self.assertRaises(AttributeError):
            prop.out_frame = ''


if __name__ == '__main__':
    unittest.main(verbosity=2)
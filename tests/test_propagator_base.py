import sys
import os
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





if __name__ == '__main__':
    unittest.main(verbosity=2)
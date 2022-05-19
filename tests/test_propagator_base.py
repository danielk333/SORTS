import sys
import os
from copy import copy
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as np
import numpy.testing as nt

from astropy.time import TimeDelta, Time

from sorts.targets import Propagator



class TestBaseProp(unittest.TestCase):

    def test_base_prop_methods(self):

        class new_propagator(Propagator):
            def propagate(self, t, state0, epoch, **kwargs):
                pass

        prop = new_propagator()
        assert prop.propagate(0,0,0) is None


    def test_convert_time(self):
        
        
        class new_propagator(Propagator):
            def propagate(self, t, state0, epoch, **kwargs):
                pass

        prop = new_propagator()

        mjd0 = 57125.7729
        epoch = Time(mjd0, format='mjd')
        dt = 10.0

        t_ref = TimeDelta(dt, format='sec')

        

        times = []
        times += [prop.convert_time(np.array([dt], dtype=np.float), epoch)] #0
        times += [prop.convert_time(dt, epoch)] #1
        times += [prop.convert_time(int(dt), epoch)] #2
        times += [prop.convert_time(TimeDelta(dt, format='sec'), epoch)] #3
        times += [prop.convert_time(epoch + TimeDelta(dt, format='sec'), epoch)] #4
        times += [prop.convert_time(epoch + TimeDelta([dt], format='sec'), epoch)] #5
        times += [prop.convert_time((epoch + TimeDelta(dt, format='sec')).datetime64, epoch)] #6
        times += [prop.convert_time((epoch + TimeDelta([dt], format='sec')).datetime64, epoch)] #7
        times += [prop.convert_time(dt, mjd0)] #8
        times += [prop.convert_time(dt, [mjd0])] #9
        times += [prop.convert_time(dt, np.array([mjd0], dtype=np.float))] #10

        for t, ep in times:
            if len(t.shape) > 0:
                t = t[0]
            nt.assert_almost_equal(t.sec, t_ref.sec, decimal=5)

            assert len(ep.shape) == 0
            assert ep == epoch


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
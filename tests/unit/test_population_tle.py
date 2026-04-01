import numpy as np
import numpy.testing as nptest
from sorts.population import tle_catalog
from sorts.propagator import Sgp4, Sgp4Settings

l1 = "1     5U 58002B   20251.29381767 +.00000045 +00000-0 +68424-4 0  9990"
l2 = "2     5 034.2510 336.1746 1845948 000.5952 359.6376 10.84867629214144"


def test_tle_pop():
    _ = tle_catalog([(l1, l2)])


def test_mean_elem_prop():
    pop = tle_catalog([(l1, l2)], save_mean_elements=True)
    spobj = pop.get_object(0)
    prop = Sgp4(Sgp4Settings(mean_elements_input=True))
    tv = np.array([0])
    state_teme0 = prop.propagate_tle(l1, l2, tv)
    state_teme1 = prop.propagate(spobj, tv)
    nptest.assert_array_almost_equal(state_teme0, state_teme1, decimal=6)


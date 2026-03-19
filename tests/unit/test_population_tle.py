import numpy as np
import numpy.testing as npt
from astropy.time import Time
from sorts.population import tle_catalog

l1 = '1     5U 58002B   20251.29381767 +.00000045 +00000-0 +68424-4 0  9990'
l2 = '2     5 034.2510 336.1746 1845948 000.5952 359.6376 10.84867629214144'

def test_tle_pop():
    pop = tle_catalog([(l1, l2)])



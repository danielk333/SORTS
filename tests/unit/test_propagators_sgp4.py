import numpy as np
import numpy.testing as nptest
from sorts.propagator import Sgp4, Sgp4Settings
from sorts import SpaceObject

l1 = "1     5U 58002B   20251.29381767 +.00000045 +00000-0 +68424-4 0  9990"
l2 = "2     5 034.2510 336.1746 1845948 000.5952 359.6376 10.84867629214144"


def test_sgp4_spobj_prop_vs_tle_prop():
    tv = np.array([0])
    prop = Sgp4(Sgp4Settings(mean_elements_input=True))
    mean_elements, B, epoch = prop.get_mean_elements(l1, l2, degrees=False)

    spobj = SpaceObject.from_kepler(
        semi_major_axis=mean_elements[0],
        eccentricity=mean_elements[1],
        inclination=mean_elements[2],
        argument_of_periapsis=mean_elements[3],
        longitude_of_ascending_node=mean_elements[4],
        mean_anomaly=mean_elements[5],
        epoch=epoch,
        frame="TEME",
        properties={"B": B},
        degrees=False,
    )

    state_teme0 = prop.propagate_tle(l1, l2, tv)
    state_teme1 = prop.propagate(spobj, tv)

    nptest.assert_array_almost_equal(state_teme0, state_teme1, decimal=6)

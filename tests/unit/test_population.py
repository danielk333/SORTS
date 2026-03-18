import numpy as np
import numpy.testing as npt
from astropy.time import Time
from sorts import Population

def make_mock_pop(num=100):
    states = np.zeros((6, num), dtype=np.float64)
    parameters = {"diameter": np.ones((num,))}
    pop = Population(
        states=states,
        epochs=Time(
            np.full((num,), 53005.0, dtype=np.float64),
            format="mjd",
            scale="utc",
        ),
        frame="TEME",
        parameters=parameters,
        object_ids=np.arange(num),
        state_format="kepler",
        anomly_type="mean",
        dtypes={"id": np.int64},
        default_dtype=np.float64,
        epoch_format="mjd",
        epoch_scale="utc",
        degrees=True,
    )
    return pop

def test_init():
    pop = make_mock_pop()

def test_copy():
    pop = make_mock_pop()
    new_pop = pop.copy()
    assert id(new_pop) != id(pop)
    assert id(new_pop.data) != id(pop.data)
    npt.assert_array_equal(new_pop.data, pop.data)

def test_filter():
    pop = make_mock_pop()
    len0 = len(pop)
    pop.filter("id", lambda idx: idx < len0//2)
    assert len(pop) == len0//2
    npt.assert_array_equal(pop.data["id"], np.arange(len0//2))


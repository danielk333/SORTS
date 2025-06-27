from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from sorts.types import Float64_as_deg, NDArray_3d, Datetime64_us


# TODO: re-eval what fields are needed
@dataclass
class Observation:
    time_range: tuple[Datetime64_us, Datetime64_us]
    """The start time and end time of the observation, inclusive on both ends"""

    # TODO: exp_num/ExperimentDetails

    snr: npt.NDArray[np.float64]

    range: npt.NDArray[np.float64]
    """2-way range in meters"""

    range_rx: npt.NDArray[np.float64]
    """1-way range in meteres"""

    # TODO: add this
    # one_way_range_rate: npt.NDArray[np.float64]
    # """1-way range rate"""

    # TODO: rename to `two_way_range_rate`
    range_rate: npt.NDArray[np.float64]
    """2-way range rate"""

    tx_k: NDArray_3d[Float64_as_deg]
    """Pointing vector in ENU in deg, from tx station to the space object"""
    rx_k: NDArray_3d[Float64_as_deg]
    """Pointing vector in ENU in deg, from rx station to the space object"""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


# TODO: re-eval what fields are needed
@dataclass
class Observation:
    snr: npt.NDArray[np.float64]
    range: npt.NDArray[np.float64]
    range_rx: npt.NDArray[np.float64]
    range_rate: npt.NDArray[np.float64]
    tx_k: npt.NDArray[np.float64]
    rx_k: npt.NDArray[np.float64]
    rcs: npt.NDArray[np.float64]


__all__ = ["Observation"]

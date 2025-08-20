import typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts.types import EnuCoordinates
from sorts.simulation_v2.passage import ExperimentPassage


DataFrameColumnName = t.Literal[
    "id",
    # columns for ExperimentPassage data
    # "passage_id" # TODO: revisit if this is needed
    "expps_experiment_detail",
    "expps_space_object",
    "expps_tx_station",
    "expps_rx_station",
    "expps_epoch",
    "expps_time_range",
    # columns for Observation data
    "snr",
    "range",
    "range_rx",
    "range_rate",
    "tx_k",
    "rx_ksnr",
    "range",
    "range_rx",
    "range_rate",
    "tx_k",
    "rx_k",
]


# TODO: re-eval what fields are needed
class Observation(t.TypedDict):
    """A TypedDict of params"""

    id: str

    # TODO: re-think this naming
    experiment_passage: ExperimentPassage

    snr: npt.NDArray[np.float64]

    range: npt.NDArray[np.float64]
    """2-way range in meters"""

    range_rx: npt.NDArray[np.float64]
    """1-way range relative to rx station in meteres"""

    # TODO: add this
    # one_way_range_rate: npt.NDArray[np.float64]
    # """1-way range rate"""

    # TODO: rename to `two_way_range_rate`
    range_rate: npt.NDArray[np.float64]
    """2-way range rate"""

    tx_k: EnuCoordinates
    """Pointing vector in ENU, from tx station to the space object"""

    rx_k: EnuCoordinates
    """Pointing vector in ENU, from rx station to the space object"""


def to_flat_dict(observation: Observation):
    d = {
        # TODO: putting normal python object in pandas df is not ideal
        **{
            f"expps_{k}": observation["experiment_passage"][k]
            for k in ExperimentPassage.__annotations__.keys()
        },
        **{
            k: observation[k]
            for k in Observation.__annotations__.keys()
            if k not in ["experiment_passage"]
        },
    }
    return d


# TODO: putting non 1-dim columns (e.g. `tx_k`, `rx_k`) in pandas df is not ideal
def list_to_dataframe(observations: list[Observation]):
    df = pd.DataFrame([to_flat_dict(obs) for obs in observations])
    return df

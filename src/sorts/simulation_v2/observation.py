from __future__ import annotations
import typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts.types import Float64_as_deg, NDArray_3x1, Datetime64_us


class DataFrameColumnNames:
    """
    Define the `pandas` `DataFrame` column names of `Observation` as class member.
    """

    # NOTE: a simple class with classmethod for iteration is used instead of
    #   `Enum` class like `class DataFrameColumnNames_(str, Enum)` for simplicity

    # TODO: add test to ensure this file up-to-date with `Schedule class

    id: t.Final = "id"
    space_object_id: t.Final = "space_object_id"
    time_range: t.Final = "time_range"
    snr: t.Final = "snr"
    range: t.Final = "range"
    range_rx: t.Final = "range_rx"
    range_rate: t.Final = "range_rate"
    tx_k: t.Final = "tx_k"
    rx_k: t.Final = "rx_k"

    @classmethod
    def all(cls) -> list[str]:
        """Return a list of all column names."""

        return [
            t.cast(str, v)
            for k, v in vars(DataFrameColumnNames).items()
            if (not k.startswith("__")) and (not isinstance(v, classmethod))
        ]


# TODO: re-eval what fields are needed
@dataclass
class Observation:
    id: str
    space_object_id: int

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

    # TODO: ENU should be a cartesian coordinate, sth seems wrong
    tx_k: NDArray_3x1[Float64_as_deg]
    """Pointing vector in ENU in deg, from tx station to the space object"""

    # TODO: ENU should be a cartesian coordinate, sth seems wrong
    rx_k: NDArray_3x1[Float64_as_deg]
    """Pointing vector in ENU in deg, from rx station to the space object"""

    @classmethod
    def list_to_dataframe(cls, observations: list[Observation]):
        df = pd.DataFrame(
            [{c: getattr(obs, c) for c in DataFrameColumnNames.all()} for obs in observations]
        )
        return df

    @property
    def dfc(self):
        """A shortcut to return the DataFrameColumnNames class"""

        return DataFrameColumnNames

    def to_dataframe(self):
        df = pd.DataFrame({c: getattr(self, c) for c in DataFrameColumnNames.all()})
        return df

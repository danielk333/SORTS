from __future__ import annotations
import typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts.types import Float64_as_deg, NDArray_3x1, Datetime64_us
from sorts.simulation_v2.passage import Passage


class DataFrameColumnNames:
    """
    Define the `pandas` `DataFrame` column names of `Observation` as class member.
    """

    # NOTE: a simple class with classmethod for iteration is used instead of
    #   `Enum` class like `class DataFrameColumnNames_(str, Enum)` for simplicity

    # TODO: add test to ensure this file up-to-date with `Schedule class

    id: t.Final = "id"

    # columns for Passage data
    # passage_id: t.Final = "passage_id" # TODO: revisit if this is needed
    space_object_id: t.Final = "space_object_id"
    tx_station_id: t.Final = "tx_station_id"
    rx_station_id: t.Final = "rx_station_id"
    epoch: t.Final = "epoch"
    time_range: t.Final = "time_range"

    # columns for Observation data
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

    @classmethod
    def all_for_passage(cls) -> list[str]:
        """Return a list of all column names for passage."""

        return [
            cls.space_object_id,
            cls.tx_station_id,
            cls.rx_station_id,
            cls.epoch,
            cls.time_range,
        ]

    @classmethod
    def all_for_observation(cls) -> list[str]:
        """Return a list of all column names for observation."""

        return [
            cls.snr,
            cls.range,
            cls.range_rx,
            cls.range_rate,
            cls.tx_k,
            cls.rx_k,
        ]


# TODO: re-eval what fields are needed
@dataclass(kw_only=True)
class Observation:
    id: str

    passage: Passage

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
        df = pd.DataFrame([obs.to_flat_dict() for obs in observations])
        return df

    @property
    def dfc(self):
        """A shortcut to return the DataFrameColumnNames class"""

        return DataFrameColumnNames

    def to_flat_dict(self):
        d = {
            **{
                self.dfc.space_object_id: self.passage.space_object.oid,
                self.dfc.tx_station_id: self.passage.tx_station.uid,
                self.dfc.rx_station_id: self.passage.rx_station.uid,
                self.dfc.epoch: self.passage.epoch,
                self.dfc.time_range: self.passage.time_range,
            },
            **{c: getattr(self, c) for c in DataFrameColumnNames.all_for_observation()},
        }
        return d

    def to_dataframe(self):
        df = pd.DataFrame(self.to_flat_dict())
        return df

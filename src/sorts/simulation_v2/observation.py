import typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts.types import AzelCoordinates_DegM, Datetime64_us, Float64_as_m
from sorts.schedule_v2.schedule import XrDataArrayIndexer


class ObservationIndexer(t.TypedDict):
    """
    A TypedDict.
    Contains info to get a subset of entries from a `Schedule`, that corresponds to an observation.
    """

    tx_indexer: XrDataArrayIndexer
    rx_indexer: XrDataArrayIndexer


class Observation(t.TypedDict):
    """A TypedDict of params"""

    id: str

    # TODO: uncomment and update the call sites of `Observation`
    # observationIndexer: ObservationIndexer

    # TODO: re-think this naming
    # experiment_passage: ExperimentPassage

    tx_time: npt.NDArray[Datetime64_us]
    rx_time: npt.NDArray[Datetime64_us]
    """
    NOTE: if there multiple simutaneous pointings, `rx_time` will contains all the time slices.
      (becase it is calculated by filtering the full schedule by ExperimentPassage time_range)
    """
    # TODO: re-eval the purpose/necessity of having both `tx_time` and `rx_time`

    snr: npt.NDArray[np.float64]

    range: npt.NDArray[Float64_as_m]
    """2-way range in meters"""

    range_rx: npt.NDArray[Float64_as_m]
    """1-way range relative to rx station in meteres"""

    # TODO: add this
    # one_way_range_rate: npt.NDArray[np.float64]
    # """1-way range rate"""

    # TODO: rename to `two_way_range_rate`
    range_rate: npt.NDArray[np.float64]
    """2-way range rate"""

    tx_k: AzelCoordinates_DegM
    """Pointing vector in local (Az, El) coordinates, from tx station to the space object"""

    rx_k: AzelCoordinates_DegM
    """Pointing vector in local (Az, El) coordinates, from rx station to the space object"""

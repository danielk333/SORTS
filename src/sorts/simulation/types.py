"""
Shared types in this subpackage.

(Types might live in their own module instead of here if it improves readability,
and the imports can be worked around, e.g, by `if t.TYPE_CHECKING`)
"""

import typing as t
from dataclasses import dataclass
from sorts import types, radar
from sorts.space_object import SpaceObject
from sorts.interpolation import Interpolator


@dataclass(kw_only=True)
class Passage:
    """Represent a passage of a space object over the field of view of a TX-RX radar station pair."""

    space_object: SpaceObject
    tx_station: radar.Station
    rx_station: radar.Station
    epoch: types.Datetime64_us
    time_range: types.TimeRange_us
    """The start time and end time of the passage, a right-open interval"""


# TODO: can be combined with type 'Passage'?
@dataclass(kw_only=True)
class SimultaneousPassage:
    """Represent a passage of a space object over the field of view of a single TX, multiple simultaneous RX radar stations."""

    space_object: SpaceObject
    tx_station: radar.Station
    rx_stations: list[radar.Station]
    epoch: types.Datetime64_us
    time_range: types.TimeRange_us
    """The start time and end time of the passage, a right-open interval"""

    def to_passages(self) -> list[Passage]:
        passages = [
            Passage(
                space_object=self.space_object,
                tx_station=self.tx_station,
                rx_station=rx_station,
                epoch=self.epoch,
                time_range=self.time_range,
            )
            for rx_station in self.rx_stations
        ]

        return passages


SpaceObjectJacobianTuple = tuple[
    SpaceObject, SpaceObject, SpaceObject, SpaceObject, SpaceObject, SpaceObject
]
"""A tuple of 6 `SpaceObject`, the first one is the original one and the next 5 are perturbated versions."""


SpaceObjectInterpolatorJacobianTuple = tuple[
    Interpolator, Interpolator, Interpolator, Interpolator, Interpolator, Interpolator
]
"""A tuple of 6 `Interpolator`, the first one is the original one and the next 5 are perturbated versions."""

"""
Shared types in this subpackage.

(Types might live in their own module instead of here if it improves readability,
and the imports can be worked around, e.g, by `if t.TYPE_CHECKING`)

# todo: probably move this out to the other types files??
"""

from dataclasses import dataclass
from sorts import types, radar
from sorts.space_object import SpaceObject


# TODO: can be combined with type 'Passage'?
@dataclass(kw_only=True)
class Passage:
    """Represent a passage of a space object over the field of view of a single TX, multiple simultaneous RX radar stations."""

    space_object: SpaceObject
    tx_station: radar.Station
    rx_stations: list[radar.Station]
    epoch: types.Datetime64_us
    time_range: types.TimeRange_us
    """The start time and end time of the passage, a right-open interval"""


SpaceObjectJacobianTuple = tuple[
    SpaceObject, SpaceObject, SpaceObject, SpaceObject, SpaceObject, SpaceObject, SpaceObject
]
"""A tuple of 7 `SpaceObject`, the first one is the original one and the next 6 are perturbated versions."""

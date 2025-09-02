"""
Shared types in this subpackages.

(Types might live in their own module instead of here if it improves readability,
and the imports can be worked around, e.g, by `if t.TYPE_CHECKING`)
"""

import typing as t
from sorts.types import Datetime64_us
from sorts.radar import Station
from sorts.space_object import SpaceObject


class Passage(t.TypedDict):
    """
    A TypedDict of params. Represent a passage of a space object over the field of view of a TX-RX radar station pair.
    """

    # id: int # TODO: revisit if this is needed
    # TODO: add ENU and/or ECEF states?

    space_object: SpaceObject
    tx_station: Station
    rx_station: Station
    epoch: Datetime64_us
    time_range: tuple[Datetime64_us, Datetime64_us]
    """The start time and end time of the passage, a right-open interval"""

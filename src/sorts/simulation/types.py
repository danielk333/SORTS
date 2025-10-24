"""
Shared types in this subpackage.

(Types might live in their own module instead of here if it improves readability,
and the imports can be worked around, e.g, by `if t.TYPE_CHECKING`)
"""

import typing as t
from sorts import types, radar, space_object


# TODO: replace objects like `SpaceObject`, `Station` by ids?
class Passage(t.TypedDict):
    """
    A TypedDict of params. Represent a passage of a space object over the field of view of a TX-RX radar station pair.
    """

    space_object: space_object.SpaceObject
    tx_station: radar.Station
    rx_station: radar.Station
    epoch: types.Datetime64_us
    time_range: types.TimeRange_us
    """The start time and end time of the passage, a right-open interval"""

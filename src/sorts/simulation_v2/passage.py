from dataclasses import dataclass
from sorts.types import Datetime64_us
from sorts.radar.tx_rx import Station
from sorts.space_object import SpaceObject


@dataclass(kw_only=True)
class Passage:
    # id: int # TODO: revisit if this is needed
    space_object: SpaceObject
    tx_station: Station
    rx_station: Station
    time_range: tuple[Datetime64_us, Datetime64_us]
    """The start time and end time of the passage, inclusive on both ends"""

import typing as t


# TODO: relocate and integrate with `Radar` class
RadarCompositeKey = tuple[str, ...]
"""
a structed unique identifer for a radar.

currently the tuple corresponds to the first 2 param of `radars.get_radar()`,
with slightly tightened typing

WIP

TODO: discuss with Daniel if tuple of strings are good enough ids
"""

RadarStationCompositeKey = tuple[str, ...]
"""
a structed unique identifer for a station of a radar.

similar to `RadarCompositeKey` but for radar station

WIP

TODO: discuss with Daniel if tuple of strings are good enough ids
"""

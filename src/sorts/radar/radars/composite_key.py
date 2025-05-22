import typing as t


# TODO: remove?
RadarCompositeKey = tuple[str, ...]
"""
a structed unique identifer for a radar.

currently the tuple corresponds to the first 2 param of `radars.get_radar()`,
with slightly tightened typing
"""

# TODO: remove?
RadarStationCompositeKey = tuple[str, ...]
"""
a structed unique identifer for a station of a radar.

similar to `RadarCompositeKey` but for radar station
"""

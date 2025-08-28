from __future__ import annotations
import logging
import xarray as xr

logger = logging.getLogger(__name__)


# TODO: rename to just `ScheduleData` when xarray adoptation is done?
ScheduleXrds = xr.Dataset
"""An xarray `Dataset` that contains the schedule data"""

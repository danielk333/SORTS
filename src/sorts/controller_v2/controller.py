import typing as t

from .random_uniform_scans_controller import RandomUniformScansController
from .tracker_controller import TrackerController
from .scanner_controller import ScannerController
from .fence_scan_controller import FenceScanController


Controller = t.Union[
    RandomUniformScansController,
    TrackerController,
    ScannerController,
    FenceScanController,
]

__all__ = ["Controller"]

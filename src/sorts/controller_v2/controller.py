import typing as t

from .random_uniform_scans_controller import RandomUniformScansController
from .fence_scan_controller import FenceScanController


# TODO: remove?
Controller = t.Union[
    RandomUniformScansController,
    FenceScanController,
]

__all__ = ["Controller"]

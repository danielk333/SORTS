from datetime import datetime
import pandas as pd
from .. import scheduler_v2 as schr
from .dumb_scheduler import DumbScheduler
from ..controller_v2 import RandomUniformScansController


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def test_SimpleScheduler():
    stt_tstmp = datetime.fromisoformat("2025-01-01 00:00:00")
    end_tstmp = datetime.fromisoformat("2025-01-02 00:00:00")
    result = DumbScheduler(
        controllers=(
            RandomUniformScansController(min_elevation_deg=75, npoints=10),
            RandomUniformScansController(min_elevation_deg=5, npoints=5),
        )
    ).generate_schedule(stt_tstmp, end_tstmp)

    pd.set_option("display.max_rows", None)  # `None` means displaying all rows
    pd.set_option("display.max_columns", None)  # `None` means displaying all columns
    print(result)

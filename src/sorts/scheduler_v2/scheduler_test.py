from .schedule_metadata import schedule_nprecord_dtype
from .scheduler import Scheduler


def test_schedule_column_configs():
    print(schedule_nprecord_dtype)


def test_Scheduler():
    Scheduler().generate_schedule()

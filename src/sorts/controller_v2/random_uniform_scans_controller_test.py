from datetime import datetime, timezone, timedelta
from sorts.schedule_v2 import ScheduleNdarrayDict2
from sorts.controller_v2.random_uniform_scans_controller import RandomUniformScansController


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def random_uniform_scans_controller_smoke_test():
    controller = RandomUniformScansController(
        radar_station_composite_key=("dummy", "0"), npoints=10
    )

    stt_tstmp = datetime.now(timezone.utc)
    end_tstmp = stt_tstmp + timedelta(hours=24)
    result = controller.generate(stt_tstmp, end_tstmp)

    for k in result:
        assert isinstance(result[k], ScheduleNdarrayDict2)

    return

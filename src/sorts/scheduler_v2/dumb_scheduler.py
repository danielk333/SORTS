import logging, typing as t
from dataclasses import dataclass, fields
from datetime import datetime
import pandas as pd
from sorts.radar.radars.composite_key import RadarStationCompositeKey
from sorts.schedule_v2 import Schedule
from sorts.controller_v2 import ControllerProtocol

logger = logging.getLogger(__name__)


# TODO: need some rework to get it working with `Simulation` and `DetectionConfig` class
@dataclass(kw_only=True)
class DumbScheduler:
    """
    invoke generate on each controllers, then blindly merge them together
    (so 0th controller's schedule is subjected to override by 1st controller, etc.)
    """

    controllers: tuple[ControllerProtocol, ...] = ()
    res_us = 1000
    "time resolution in microseconds. defaults to `1000` (1ms)"

    # TODO: make it work with dict of schedule
    def generate_schedule(
        self, stt_tstmp: datetime, end_tstmp: datetime
    ) -> dict[RadarStationCompositeKey, Schedule]:
        """
        Takes start and end time and returns a `Schedule`.
        """

        sch_field_names = [f.name for f in fields(Schedule)]
        merged_sch_df = pd.DataFrame(columns=sch_field_names).set_index("stt_tstmp_us", drop=False)

        for controller in self.controllers:
            ctrlr_sch = controller.generate(stt_tstmp, end_tstmp, self.res_us)
            ctrlr_df = pd.DataFrame({f: getattr(ctrlr_sch, f) for f in sch_field_names}).set_index(
                "stt_tstmp_us", drop=False
            )

            merged_sch_df = merged_sch_df.reindex(merged_sch_df.index.union(ctrlr_df.index))
            merged_sch_df.update(ctrlr_df)

        merged_sch = Schedule(
            **{col: merged_sch_df[col].to_numpy() for col in merged_sch_df.columns}
        )

        return merged_sch

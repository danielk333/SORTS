import logging, typing as t, abc
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from .. import scheduler_v2 as schr
from .. import controller_v2 as ctrlr


@dataclass(kw_only=True)
class DumbScheduler(schr.SchedulerProtocol):
    """
    invoke generate on each controllers, then blindly merge them together
    (so 0th controller's schedule is subjected to override by 1st controller, etc.)
    """

    controllers: tuple[ctrlr.ControllerProtocol, ...] = ()
    res_us = 1000
    "time resolution in microseconds. defaults to `1000` (1ms)"

    def generate_schedule(self, stt_tstmp: datetime, end_tstmp: datetime) -> schr.Schedule:
        """
        Takes start and end time and returns a `Schedule`.
        """

        ret_df = pd.DataFrame(columns=[*schr.schedule_column_names.values()]).set_index(
            schr.schedule_column_names["stt_tstmp"]
        )
        ret_df.index = pd.to_datetime(ret_df.index)
        for controller in self.controllers:
            ctrlr_df = controller.generate(stt_tstmp, end_tstmp, self.res_us)
            ret_df = ret_df.reindex(ret_df.index.union(ctrlr_df.index))
            ret_df.update(ctrlr_df)

        return ret_df

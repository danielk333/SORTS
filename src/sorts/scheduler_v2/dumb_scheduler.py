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
    res_ns = int(1e6)
    "time resolution in nanoseconds. defaults to `1e6` (1ms)"

    def generate_schedule(self, stt_tstmp: datetime, end_tstmp: datetime) -> pd.DataFrame:
        """
        Takes times and a corresponding generator that returns radar instances to generate a radar schedule.

        Returns
        ---
        a DataFrame with these columns:

        |index     |coordinate_system|coh_int_bandwidth|pointing                 |ipp    |pulse_length|
        |:-        |:-               |:-               |:-                       |:-     |:-          |
        |datetime64|int8             |float64          |(float64,float64,float64)|float64|float64     |
        """

        # ret_df = pd.DataFrame(
        #     columns=[*schr.schedule_column_names.values()], index=pd.to_datetime([])
        # )
        ret_df = pd.DataFrame(columns=[*schr.schedule_column_names.values()]).set_index(
            schr.schedule_column_names["stt_tstmp"]
        )
        ret_df.index = pd.to_datetime(ret_df.index)
        # ret_df = pd.DataFrame(columns=[], index=pd.to_datetime([]))
        for controller in self.controllers:
            ctrlr_df = controller.generate(stt_tstmp, end_tstmp, self.res_ns)
            ret_df = ret_df.reindex(ret_df.index.union(ctrlr_df.index))
            ret_df.update(ctrlr_df)

            # ret_df = ret_df.merge(
            #     controller.generate(stt_tstmp, end_tstmp, self.res_ns),
            #     how="outer",
            #     left_index=True,
            #     right_index=True,
            #     # suffixes=("_x", None),
            # )
        # ctrlr_df = controller.generate(stt_tstmp, end_tstmp, self.res_ns)
        # ret_df = ret_df.reindex(ret_df.index.union(ctrlr_df.index))
        # ret_df.update()

        # ret_df = ret_df.join(
        #     [c.generate(stt_tstmp, end_tstmp, self.res_ns) for c in self.controllers],
        #     how="outer",
        # )

        return ret_df

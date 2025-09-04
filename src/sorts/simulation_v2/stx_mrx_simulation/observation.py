from sorts.types import TxRxTuple
from sorts.schedule import XrDataArrayIndexer, Schedule
from sorts.simulation_v2.types import Passage
from .simulation_unit import SimulationUnit, StateData


_SK = Schedule._K
_SuK = SimulationUnit._K

# TODO: rename to sth like `IndexerOverSchedule`
ObservationIndexer = TxRxTuple[XrDataArrayIndexer, XrDataArrayIndexer]
"""
Should be used with `Passage`.
Can be used to get a subset of entries from a `Schedule`, that corresponds to an observation.
"""


class Observation:
    def __init__(
        self,
        passage: Passage,
        indexer: ObservationIndexer,
        simulation_unit: SimulationUnit,
    ):
        self.passage = passage
        self.indexer = indexer
        self.simulation_unit = simulation_unit

    def get_schedule_slice(self) -> TxRxTuple[Schedule, Schedule]:
        """Returns subset of schedules, in `(tx_scheule, tx_schedule` that corresponds to the observation"""

        tx_sch_ps = self.simulation_unit.tx_schedule.filter_by_time_range(
            self.passage["time_range"]
        )
        rx_sch_ps = self.simulation_unit.rx_schedule.filter_by_time_range(
            self.passage["time_range"]
        )

        tx_sch_obs = Schedule(data=tx_sch_ps._data.loc[{_SK.start_time: self.indexer.tx}])
        rx_sch_obs = Schedule(data=rx_sch_ps._data.loc[{_SK.start_time: self.indexer.rx}])

        return TxRxTuple(tx=tx_sch_obs, rx=rx_sch_obs)

    # TODO: add test?
    def get_state_slice(self) -> StateData:
        """Get the subset of `simulation_unit.StateData` data the corresponds to the the observation"""

        # TODO: a more robust way is preferred.
        #   right now it works by assuming the simulation_unit consist of non-overlapping passages,
        #   and thus indexer over a schedule filtered by a passage
        #   will also work on simulation unit data filtered by the same passage
        time_mask = (
            # __forcing_line_break__
            (self.simulation_unit._state_data[_SuK.time] >= self.passage["time_range"][0])
            & (self.simulation_unit._state_data[_SuK.time] <= self.passage["time_range"][1])
        )

        sim_state_slice = self.simulation_unit._state_data.loc[{_SuK.time: time_mask}].loc[
            {_SuK.time: self.indexer.rx.to_numpy()}
        ]

        return StateData(sim_state_slice)

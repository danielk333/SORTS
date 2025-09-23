from sorts.types import TxRxTuple
from sorts.schedule import XrDataArrayIndexer, Schedule
from sorts.simulation.types import Passage
from .simulation_unit import SimulationUnit, StateData


_SK = Schedule._K
_SuK = SimulationUnit._K

# TODO: rename to sth like `IndexerOverSchedule`
ObservationIndexer = TxRxTuple[XrDataArrayIndexer, XrDataArrayIndexer]
"""
Should be used with `Passage`.
Can be used to get a subset of entries from a `Schedule`, that corresponds to an observation.
"""


# TODO: param tx_schedule, rx_schedule are added as tmp solution, we should have a Dto/Serializable ObservationIndexer type
class Observation:
    def __init__(
        self,
        passage: Passage,
        indexer: ObservationIndexer,
        simulation_unit: SimulationUnit,
        tx_schedule: Schedule,
        rx_schedule: Schedule,
    ):
        self.passage = passage
        self.indexer = indexer
        self.simulation_unit = simulation_unit
        self.tx_schedule = tx_schedule
        self.rx_schedule = rx_schedule

    def get_schedule_slice(self) -> TxRxTuple[Schedule, Schedule]:
        """Returns subset of schedules, in `(tx_scheule, tx_schedule` that corresponds to the observation"""

        tx_sch_obs = Schedule(data=self.tx_schedule._data.loc[{_SK.multi_index: self.indexer.tx}])
        rx_sch_obs = Schedule(data=self.rx_schedule._data.loc[{_SK.multi_index: self.indexer.rx}])

        return TxRxTuple(tx=tx_sch_obs, rx=rx_sch_obs)

    def get_state_slice(self) -> StateData:
        """Get the subset of `simulation_unit.StateData` data the corresponds to the the observation"""

        # TODO: maybe a mutating the index label is here is better than `.to_numpy().tolist()`?
        # NOTE: `_state_data`'s `multi_index` has different label than `ScheduleData`'s `multi_index`
        #   but content-wise they are the same thing, so `self.indexer.rx.to_numpy().tolist()` is used a shortcut here
        sim_state_slice = self.simulation_unit._state_data.loc[
            {_SuK.multi_index: self.indexer.rx.to_numpy().tolist()}
        ]

        return StateData(sim_state_slice)

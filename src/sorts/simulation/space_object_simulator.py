import dataclasses, typing as t
import numpy as np
import numpy.typing as npt
from sorts import types, utils, schedule, passage
from sorts.types import Datetime64_us, EcefStates, Datetime_Like, Float64_as_sec
from sorts.utils import to_datetime64_us
from sorts.interpolation import Interpolation
from sorts.propagator import Propagator
from sorts.space_object import SpaceObject
from sorts.radar import Station, StationId
from . import tx_rx_pair_state


@dataclasses.dataclass(kw_only=True, frozen=True)
class SpaceObjectSimulator:
    """A convenience object for simulation of a `SpaceObject`."""

    propagator: Propagator
    interpolator: t.Type[Interpolation]

    def propagate(
        self,
        space_object: SpaceObject,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        time_step: float,
    ) -> tuple[npt.NDArray[Float64_as_sec], EcefStates]:
        """Propagate the states of a space object."""

        start_time = to_datetime64_us(start_time)
        end_time = to_datetime64_us(end_time)

        dt = (end_time - start_time) / np.timedelta64(1, "s")
        t0 = (start_time - to_datetime64_us(space_object.epoch)) / np.timedelta64(1, "s")
        times = np.arange(t0, t0 + dt, time_step, dtype=np.float64)

        itrs_states = self.propagator.propagate(space_object, times)

        return times, itrs_states

    def make_interpolation(
        self,
        times: npt.NDArray[Float64_as_sec],
        states: EcefStates,
    ) -> Interpolation:
        return self.interpolator(states=states, t=times)

    def simulate(
        self,
        space_object: SpaceObject,
        states_interpolation: Interpolation,
        passages: list[passage.Passage],
        sch: schedule.ScheduleDataframe,
        station_map: t.Mapping[StationId, Station],
        exp_detail_map: types.ExperimentDetailMap,
    ) -> list[tx_rx_pair_state.TxRxPairState]:
        """
        Run a simulation for the space object using the provided propagation, over the specified passages.
        """

        _K = tx_rx_pair_state.TxRxPairStateKey

        epoch = utils.to_datetime64_us(space_object.epoch)
        spobj_diameter = space_object.d
        # TODO: confirm with daniel if setting a default radar_albedo is okay
        spobj_radar_albedo = space_object.properties.get("radar_albedo", 1.0)

        txrx_state_dict = tx_rx_pair_state.gather_from_passages_schedule(
            passages=passages,
            sch=sch,
        )

        sim_result: list[tx_rx_pair_state.TxRxPairState] = []
        for stn_id_pair, txrx_state in txrx_state_dict.items():
            dsec = (txrx_state[_K.time].to_numpy() - epoch) / np.timedelta64(1, "s")
            spobj_state = states_interpolation.get_state(dsec)

            sim_result.append(
                tx_rx_pair_state.simulate(
                    txrx_state=txrx_state,
                    spobj_state=spobj_state,
                    spobj_diameter=spobj_diameter,
                    spobj_radar_albedo=spobj_radar_albedo,
                    tx_station=station_map[stn_id_pair[0]],
                    rx_station=station_map[stn_id_pair[1]],
                    exp_detail_map=exp_detail_map,
                )
            )

        return sim_result

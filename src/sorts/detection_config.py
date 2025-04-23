import typing as t
from dataclasses import dataclass, fields
from datetime import datetime
import numpy as np
import numpy.typing as npt
from .schedule_v2 import Schedule
import sorts
from sorts.radar.radar import Radar
from sorts.radar.radars.composite_key import RadarStationCompositeKey
from sorts.radar.tx_rx import Station
from sorts.passes_v2 import Pass
from sorts.calculations import Observation, ExperimentDetail


@dataclass
class SimpleStxSrx:
    stt_tstmp_us: npt.NDArray[np.datetime64]
    "the timestamp which observations can be made"

    tx_station: Station
    rx_station: Station

    tx_schedule: Schedule
    "the tx radar station schedule, can contain more entries than `stt_tstmp_us`"
    rx_schedule: Schedule
    "the rx radar station schedule, can contain more entries than `stt_tstmp_us`"

    exp_num_map: dict[int, ExperimentDetail]


@dataclass
class StxMrx(t.NamedTuple):
    """wip, do not use it"""

    stt_tstmp_us: npt.NDArray[np.datetime64]
    tx_station_key: RadarStationCompositeKey
    rx_station_keys: list[RadarStationCompositeKey]
    tx_schedule: Schedule
    rx_schedules: list[Schedule]
    exp_num_map: dict[int, ExperimentDetail]


DetectionConfig: t.TypeAlias = t.Union[SimpleStxSrx, StxMrx]


# TODO: should move to other file? e.g. calculations?
# TODO: should vectorize and work against an array of SpaceObject?
def calculate_simple_stx_srx_observations(
    dcfg: SimpleStxSrx, space_object: sorts.SpaceObject, epoch: datetime
) -> Observation:
    dt_arr: npt.NDArray[np.float64] = (
        (dcfg.stt_tstmp_us - np.datetime64(epoch)).astype("timedelta64[us]").astype(np.float64)
    )
    states = space_object.get_state(dt_arr)

    space_object_tx_enu = dcfg.tx_station.enu(states)  # space object in tx station coordinate
    space_object_rx_enu = dcfg.rx_station.enu(states)  # space object in rx station coordinate

    range_tx_m = np.linalg.norm(space_object_tx_enu[:3, :], axis=0)
    range_rx_m = np.linalg.norm(space_object_rx_enu[:3, :], axis=0)

    snr = np.empty((len(dcfg.stt_tstmp_us),), dtype=np.float64)
    rcs = np.empty((len(dcfg.stt_tstmp_us),), dtype=np.float64)

    powers = np.empty((len(dcfg.stt_tstmp_us),), dtype=np.float64)
    t_slices = np.empty((len(dcfg.stt_tstmp_us),), dtype=np.float64)
    txrx_on = np.full((len(dcfg.stt_tstmp_us),), False, dtype=bool)

    pulse_lengths = np.array(
        [dcfg.exp_num_map[n].pulse_length for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )
    ipps = np.array([dcfg.exp_num_map[n].ipp for n in dcfg.tx_schedule.exp_num], dtype=np.float64)
    powers = np.array(
        [dcfg.exp_num_map[n].power for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )
    bandwidths = np.array(
        [dcfg.exp_num_map[n].bandwidth for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )
    duty_cycles = np.array(
        [dcfg.exp_num_map[n].duty_cycle for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )
    rx_noise_temps = np.array(
        [dcfg.exp_num_map[n].noise_temp for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )

    # TODO: `pointing` and `vectorized_parameters` seems not needed?
    #   pointing can be obtained by [dcfg.tx_schedule.pointing_az, dcfg.tx_schedule.pointing_el, range?]
    tx_gain_arr = dcfg.tx_station.beam.gain(
        space_object_tx_enu[:3],
        # pointing=vectorized_data[:, 0:3].T,
        # vectorized_parameters=True,
    )
    rx_gain_arr = dcfg.rx_station.beam.gain(
        space_object_rx_enu[:3],
        # pointing=vectorized_data[:, 4:7].T,
        # vectorized_parameters=True,
    )

    tx_wavelength: float = dcfg.tx_station.beam.wavelength
    rx_wavelength: float = dcfg.rx_station.beam.wavelength

    # TODO: complete `hard_target_snr` support
    # snr = sorts.signals.hard_target_snr(
    #     tx_gain_arr,
    #     rx_gain_arr,
    #     tx_wavelength,
    #     powers,
    #     range_tx_m,
    #     range_rx_m,
    #     diameter=space_object.d,
    #     bandwidth=bandwidths[0],
    #     rx_noise_temp=rx_noise_temps[0],
    #     radar_albedo=space_object.parameters.get("radar_albedo", 1.0),
    # )

    # TODO: add `doppler_spread_integrated_snr:` support
    # TODO: add `blind_ranges:` support

    obs = Observation(
        snr=np.full(len(dcfg.stt_tstmp_us), 1.0, dtype=np.float64),  # TODO: imple
        range=range_tx_m + range_rx_m,
        range_rx=range_rx_m,
        range_rate=np.full(len(dcfg.stt_tstmp_us), 1.0, dtype=np.float64),  # TODO: imple
        tx_k=space_object_tx_enu[:3] / range_tx_m,
        rx_k=space_object_rx_enu[:3] / range_rx_m,
        rcs=np.full(len(dcfg.stt_tstmp_us), 1.0, dtype=np.float64),  # TODO: imple
    )

    return obs

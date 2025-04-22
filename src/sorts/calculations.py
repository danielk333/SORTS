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

# from sorts.radar.radar_v2 import Radar # TODO: imple and use Radar v2

vectorized_row_shape = (8,)  # TODO: this is a tmp soution


# TODO: this is a tmp soution
# TODO: maybe need to support cases where some of them varies by time?
@dataclass
class ExperimentDetail:
    coh_int_bandwidth: float
    ipp: float
    pulse_length: float
    power: float
    bandwidth: float
    duty_cycle: float
    noise_temp: float


# TODO: this is a tmp soution
exp_num_map: dict[int, ExperimentDetail] = {0: ExperimentDetail(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)}


@dataclass
class Observation:
    snr: npt.NDArray[np.float64]
    range: npt.NDArray[np.float64]
    range_rx: npt.NDArray[np.float64]
    range_rate: npt.NDArray[np.float64]
    tx_k: npt.NDArray[np.float64]
    rx_k: npt.NDArray[np.float64]
    rcs: npt.NDArray[np.float64]


def calculate_observation(
    tx_station_key: RadarStationCompositeKey,
    rx_station_key: RadarStationCompositeKey,
    tx_schedule: Schedule,
    rx_schedule: Schedule,
    space_object: sorts.SpaceObject,
    # tstmp: datetime,
    # interpolator=None, # TODO: add interpolator support?
) -> Observation:
    """
    calculate observations for a tx-rx station pair and an object population

    based on `src/sorts/scheduler/observed_parameters.py` -> `calculate_observation()` atm

    wip

    NOTE: only rows of `tx_schedule` and `rx_scedule` which has the same `stt_tstmp_us` will be calculated
    """

    # if not np.array_equal(tx_schedule.stt_tstmp_us, tx_schedule.stt_tstmp_us):
    #     raise RuntimeError(f"")

    tx_station: Station = get_station_by_key(tx_station_key)
    rx_station: Station = get_station_by_key(rx_station_key)

    diam = space_object.d
    spin_period = space_object.parameters.get("spin_period", None)
    radar_albedo = space_object.parameters.get("radar_albedo", 1.0)

    # NOTE:
    #   - we keep only the rows with same `stt_tstmp_us` in both tx and rx schedule
    #   - then we convert it to ndarray of float64 representing delta time in seconds
    dt_arr: npt.NDArray[np.float64] = (
        np.intersect1d(tx_schedule.stt_tstmp_us, rx_schedule.stt_tstmp_us)
        .astype("timedelta64[us]")
        .astype(np.float64)
    )

    states = space_object.get_state(dt_arr)

    snr = np.empty((len(tx_schedule.stt_tstmp_us),), dtype=np.float64)
    snr_inch = np.empty((len(tx_schedule.stt_tstmp_us),), dtype=np.float64)
    rcs = np.empty((len(tx_schedule.stt_tstmp_us),), dtype=np.float64)
    keep = np.full((len(tx_schedule.stt_tstmp_us),), True, dtype=bool)

    enus = [
        tx_station.enu(states),
        rx_station.enu(states),
    ]
    ranges = [np.linalg.norm(enu[:3, :], axis=0) for enu in enus]
    range_rates = [
        np.sum(enu[3:, :] * (enu[:3, :] / np.linalg.norm(enu[:3, :], axis=0)), axis=0)
        for enu in enus
    ]

    # TODO: eval if we need need to keep it
    # metas = []

    powers = np.empty((len(dt_arr),), dtype=np.float64)
    pulse_lengths = np.full(
        len(tx_schedule.stt_tstmp_us),
        exp_num_map[int(tx_schedule.exp_num)].pulse_length,
        dtype=np.float64,
    )  # TODO: tx_schedule.pulse_length?
    ipps = np.full(
        len(tx_schedule.stt_tstmp_us), exp_num_map[int(tx_schedule.exp_num)].ipp, dtype=np.float64
    )  # TODO: tx_schedule.coh_int_bandwidth?
    bandwidths = np.full(
        len(tx_schedule.stt_tstmp_us),
        exp_num_map[int(tx_schedule.exp_num)].coh_int_bandwidth,
        dtype=np.float64,
    )  # TODO: tx_schedule.coh_int_bandwidth?
    duty_cycles = np.empty((len(dt_arr),), dtype=np.float64)
    rx_noise_temps = np.empty((len(dt_arr),), dtype=np.float64)
    txrx_on = np.full((len(dt_arr),), False, dtype=bool)

    vectorized_data = np.empty((len(dt_arr), *vectorized_row_shape), dtype=np.float64)
    # TODO: eval if we still need to support `stop_condition`
    # TODO: eval if we still need to support `extended_meta`
    # TODO: loop over `for ri, (radar, meta) in enumerate(generator)`?

    vec_row = get_vectorized_row(tx_station, rx_station)

    vectorized_data[ri, :] = vec_row

    # TODO: eval if we need need to keep it
    # t_slices = np.empty((len(dt_arr),), dtype=np.float64)
    # t_slice_ = meta.get("t_slice", None)
    # if t_slice_ is not None:
    #     t_slices[ri] = t_slice_
    # else:
    #     t_slices[ri] = np.nan

    pulse_lengths[ri] = tx_station.pulse_length
    ipps[ri] = tx_station.ipp
    powers[ri] = tx_station.power
    bandwidths[ri] = tx_station.coh_int_bandwidth
    duty_cycles[ri] = tx_station.duty_cycle
    rx_noise_temps[ri] = rx_station.noise

    # TODO: eval if we need need to keep these?
    # if radar.tx[txi].enabled and radar.rx[rxi].enabled:
    #     txrx_on[ri] = True
    # observable = np.full(dt_arr.shape, True, dtype=bool)
    # keep = np.logical_and(observable, txrx_on)

    snr_modulation = 1.0
    if blind_ranges:
        # check if target is in radars blind range
        # assume synchronized transmitting
        # assume decoding of partial pulses is possible and linearly decreases signal strength
        for ch_txi, ch_rxi in radar.joint_stations:
            if ch_rxi == rxi:
                delay = (ranges[0][ti] + ranges[1][ti]) / scipy.constants.c
                ipp_f = np.mod(delay, radar.tx[txi].ipp)

                if ipp_f <= radar.tx[txi].pulse_length:
                    snr_modulation = ipp_f / radar.tx[txi].pulse_length
                elif ipp_f >= radar.tx[txi].ipp - radar.tx[txi].pulse_length:
                    snr_modulation = (radar.tx[txi].ipp - ipp_f) / radar.tx[txi].pulse_length
                break

    tx_g, tx_wavelength = self.get_beam_gain_and_wavelength(
        radar.tx[txi].beam,
        enus[0][:3, ti],
        meta,
    )
    rx_g, rx_wavelength = self.get_beam_gain_and_wavelength(
        radar.rx[rxi].beam,
        enus[1][:3, ti],
        meta,
    )

    obs = Observation(
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    return obs


def get_station_by_key(key: RadarStationCompositeKey) -> Station:
    """a tmp 'get_station_by_key()' implementation which only support 2 hard-coded keys"""

    eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")

    match key:
        case ("eiscat3d", "stage1-array", "tx", "0"):
            station = eiscat3d.tx[0]
        case ("eiscat3d", "stage1-array", "rx", "1"):
            station = eiscat3d.rx[1]
        case _:
            raise RuntimeError(f"unsupported radar station key: {key}")

    return station


# TODO: re-eval. feels like pandas/dataframes work here?
def get_vectorized_row(tx_station: Station, rx_station: Station):
    """Used to extract the used data from the `radar` instance when vectorizing `calculate_observation`.
    Input to `vectorized_observable_filter` and `vectorized_get_beam_gain_and_wavelength`.

    Should return a numpy vector, will be stored as a row-vector in the matrix passed to the vectorized functions.
    """

    row = np.empty(vectorized_row_shape, dtype=np.float64)
    row[0:3] = tx_station.beam.pointing[:]
    row[3] = tx_station.beam.wavelength
    row[4:7] = rx_station.beam.pointing[:]
    row[7] = rx_station.beam.wavelength

    return row

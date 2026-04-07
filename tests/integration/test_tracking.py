"""
We test the integration of the following modules in this test:
- space_object
- ...

Notably, we used the space object state directly here and `interpolation` module is not involved/tested.

We check against an imaginary circular earth orbit which
- co-rotate with Earth (i.e. a circular orbit on a earth fixed coordinate system)
- at 90deg inclination, 0 deg longitude
- simulation start from the orbit's intersection with earth's equatorial plane, passing south hemisphere then north hemisphere

The radar station is also intentionally positioned at (lat=0, long=0),
and we abuse coordinate frames by using GCRS coordinates directly as if it is ECEF coordinates,
we so that:
- its local east axis align with the earth's
- its local north axis align with the earth's north
- its local up axis
"""

# TODO: complete the description/explanation of the setup; maybe add a picture for the orbit?

import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time
from astropy.constants import R_earth  # type: ignore
import pyant
from sorts import (
    types,
    utils,
    schedule,
    pointing,
    simulation,
    radar,
    propagator,
    passage,
    SpaceObject,
)
from sorts.simulation import tx_rx_pair_state


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ref: https://docs.pytest.org/en/stable/how-to/xunit_setup.html#method-and-function-level-setup-teardown
def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def test_south_to_north_circular_orbit():
    spobj_orbital_radius = 7000e3  # in meters
    num_prop_steps = 60
    float_equality_thld = 1e-9

    start_time = Time("2025-01-01T02:45:00", format="isot", scale="utc")
    start_time_dt64 = utils.to_datetime64_us(start_time)

    # we make the orbit co-rotate with earth by abusing coordinate frames.
    # we generate space states on GCRS frame and directly use it as if it is on ECEF frame.
    spobj = SpaceObject.from_kepler(
        semi_major_axis=spobj_orbital_radius,
        eccentricity=0,
        inclination=90,
        argument_of_periapsis=0,
        longitude_of_ascending_node=0,
        mean_anomaly=180,
        epoch=start_time,
        frame="GCRS",
        properties={"d": 1.0},  # diameter of the spobj
        degrees=True,
    )

    spobj_delta_secs: npt.NDArray[types.Float64_as_sec] = np.linspace(
        0, spobj.state.period[0], num_prop_steps + 1
    )
    spobj_abs_times: npt.NDArray[types.Datetime64_us] = (
        spobj_delta_secs * np.timedelta64(1, "s") + start_time_dt64
    )

    spobj_state: types.EcefStates = propagator.Kepler(propagator.KeplerSettings()).propagate(
        spobj, spobj_delta_secs
    )

    tx_stn = radar.Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=pyant.models.Isotropic(),
        frequency=233e6,  # same as eisat_3d
        beam_parameters=pyant.models.IsotropicParams(),
        uid=0,
    )

    rx_stn = radar.Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=pyant.models.Isotropic(),
        frequency=233e6,  # same as eisat_3d
        beam_parameters=pyant.models.IsotropicParams(),
        uid=1,
    )

    passages = passage.find_simultaneous_passages(
        dt=spobj_delta_secs,
        space_object=spobj,
        states=spobj_state[:3, ...],
        tx_station=tx_stn,
        rx_stations=[rx_stn],
        epoch=start_time_dt64,
    )

    exp_detail = types.ExperimentDetail(
        id=0,
        coh_int_bandwidth=1.0,
        ipp=1.0,
        pulse_length=1.0,
        power=5000000.0,
        bandwidth=52.08333333333333,
        duty_cycle=1.0,
        noise_temp=150.0,
        slice_duration=np.timedelta64(10_000, "us"),  # 10ms
    )

    tracker_sch = pointing.tracking(
        spobj_ecef_states=spobj_state,
        spobj_ecef_states_times=spobj_abs_times,
        tx_station=tx_stn,
        rx_stations=[rx_stn],
        exp_id=exp_detail.id,
        slice_duration=exp_detail.slice_duration,
    )

    pairs_ls = [
        schedule.schedule_dataframe.get_tx_rx_pointing_pairs(
            sch=tracker_sch,
            start_time=ps.time_range[0],
            end_time=ps.time_range[1],
            tx_stn_num=tx_stn.uid,
            rx_stn_num=rx_stn.uid,
        )
        for ps in passages
    ]

    pairs = schedule.TxRxPointingPairs(pd.concat(pairs_ls))
    pairs = pairs.sort_values(
        by=schedule.TxRxPointingPairsKey.time, ascending=True, ignore_index=True
    )
    txrx_state = tx_rx_pair_state.from_tx_rx_pointing_pairs(pairs)

    filtered_spobj_state = spobj_state[
        :, np.isin(spobj_abs_times, txrx_state.index.get_level_values("time"))
    ]

    result_state = tx_rx_pair_state.simulate(
        txrx_state=txrx_state,
        spobj_state=filtered_spobj_state,
        spobj_diameter=spobj.d,
        spobj_radar_albedo=spobj.properties.get("radar_albedo", 1.0),
        tx_station=tx_stn,
        rx_station=rx_stn,
        exp_detail_map={exp_detail.id: exp_detail},
    )

    _K = simulation.TxRxPairStateKey

    tx_pointings_norm = np.linalg.norm(
        [
            result_state[_K.tx_pointing_e],
            result_state[_K.tx_pointing_n],
            result_state[_K.tx_pointing_u],
        ],
        axis=0,
    )
    tx_pointings_e_normalized = result_state[_K.tx_pointing_e] / tx_pointings_norm
    tx_pointings_n_normalized = result_state[_K.tx_pointing_n] / tx_pointings_norm
    tx_pointings_u_normalized = result_state[_K.tx_pointing_u] / tx_pointings_norm

    # assert `E` componend of tx pointings stayed around zero
    assert np.all(tx_pointings_e_normalized < float_equality_thld)

    # assert the propagation point at 1/2 spobj_orbital_period is in the `result_state`
    half_orbit_mask = (
        result_state.index.get_level_values(_K.time) == spobj_abs_times[int(num_prop_steps / 2)]
    )
    assert half_orbit_mask.sum() == 1  # ensure it exist and is exactly once

    # assert the normalized tx pointings at 1/2 spobj_orbital_period is (0, 0, 1)
    assert all(tx_pointings_e_normalized[half_orbit_mask] < float_equality_thld)
    assert all(tx_pointings_n_normalized[half_orbit_mask] < float_equality_thld)
    assert all(tx_pointings_u_normalized[half_orbit_mask] - 1 < float_equality_thld)

    # assert the max snr time is at half orbital period
    assert (
        spobj_abs_times[int(num_prop_steps / 2)] == t.cast(tuple, result_state[_K.snr].idxmax())[2]
    )

    # TODO: assert the start and end time of the observation is as expected? it will likely requires interpolation.
    # earth_radius: np.float64 = R_earth.value  # in meters
    # spobj_orbital_period: types.Float64_as_sec = pyorb.orbital_period(
    #         spobj_orbital_radius, pyorb.GM_earth
    #     )
    # end_time = start_time_dt64 + spobj_orbital_period * np.timedelta64(int(1e6), "us")
    #
    # passage_angular_duration: Float64_as_deg = np.degrees(
    #     np.arccos(earth_radius / spobj_orbital_radius) * 2
    # )
    # passage_duration: Float64_as_sec = passage_angular_duration / 360 * spobj_orbital_period
    # expected_passage_start_time = (
    #     start_time_dt64
    #     + (spobj_orbital_period - passage_duration) / 2 * np.timedelta64(int(1e6), "us")
    # ) # fmt: skip
    # expected_passage_end_time = expected_passage_start_time + passage_duration * np.timedelta64(
    #     int(1e6), "us"
    # )

    return

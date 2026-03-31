"""
We check against an imaginary circular orbit which
- co-rotate with Earth (i.e. a circular orbit on a earth fixed coordinate system)
- at 90deg inclination, 0 deg longitude
- simulation start from the orbit's intersection with earth's equatorial plane, passing south hemisphere then north hemisphere

we use 2 rx stations and 2 scan ranges in this setup:

for scan ranges:
- 1st range is a unrealistically low value for triggering masking effect due to min_elevation of station
- 2nd range is a normal on that is close to `spobj_orbital_radius`

for rx stations:
- 1st one is the same as tx station, and therefore should have the exact same pointings as tx station
- 2nd one is slightly offseted, with a small, non-zero min_elevation. It should produce 1 observation:
  - which points to the same location as the 2nd observation of the 1st station, with a bit of masking due to `min_elevation`
  - no observation for the 1st scan range, beccause it is out of field-of-view/range
"""

import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time
from astropy.constants import R_earth  # type: ignore
import pyant, pyorb
from sorts.types import Float64_as_sec, Float64_as_deg, Float_as_sec, Float_as_m
from sorts.frames import enu_to_ecef
from sorts import (
    types,
    utils,
    schedule,
    pointing,
    interpolation,
    simulation,
    radar,
    propagator,
    passage,
    SpaceObject,
    InterpolatedPropagation,
)
from sorts.simulation import stx_mrx_simulation, tx_rx_pair_state


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


# float_equality_thld = 1e-9
# # TODO: re-eval this threshold
# pointing_range_equality_thld: Float_as_m = 1e-6

# dt_equality_thld = control_slice_duration
# dsec_sampling_intv: Float_as_sec = 30
# scan_ranges = np.array([10, spobj_orbital_radius], dtype=np.float64)
# simu_num = len(scan_ranges)

# _SK = schedule.ScheduleKey
# _SuK = simulation.TxRxPairStateKey


def test_south_to_north_circular_orbit():
    earth_radius: np.float64 = R_earth.value  # in meters
    spobj_orbital_radius = 7000e3  # in meters
    scan_ranges = np.array([10, spobj_orbital_radius], dtype=np.float64)
    num_prop_steps = 60
    float_equality_thld = 1e-9

    spobj_orbital_period: Float64_as_sec = pyorb.orbital_period(
        spobj_orbital_radius, pyorb.GM_earth
    )

    passage_angular_duration: Float64_as_deg = np.degrees(
        np.arccos(earth_radius / spobj_orbital_radius) * 2
    )
    passage_duration: Float64_as_sec = passage_angular_duration / 360 * spobj_orbital_period

    start_time = Time("2025-01-01T02:45:00", format="isot", scale="utc")  # simulation start time
    start_time_dt64 = utils.to_datetime64_us(start_time)
    expected_passage_start_time = start_time_dt64 + (
        spobj_orbital_period - passage_duration
    ) / 2 * np.timedelta64(int(1e6), "us")
    expected_passage_end_time = expected_passage_start_time + passage_duration * np.timedelta64(
        int(1e6), "us"
    )
    end_time = start_time_dt64 + spobj_orbital_period * np.timedelta64(
        int(1e6), "us"
    )  # simulation end time

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

    spobj_state: types.EcefStates = propagator.Kepler(propagator.KeplerSettings()).propagate(
        spobj, spobj_delta_secs
    )

    # NOTE: all stations are physically the same but just have a different id;
    #   this is done so that we can retain both tx and rx pointings in the schedule,
    #   as `pointing.fence_scanning` will drop the rx entries if both tx and rx use the same station.
    tx_stn, *rx_stns = [
        radar.Station(
            lat=0.0,
            lon=0.0,
            alt=0.0,
            min_elevation=0.0,
            beam=pyant.models.Isotropic(),
            frequency=233e6,  # same as eisat_3d
            beam_parameters=pyant.models.IsotropicParams(),
            uid=idx,
        )
        for idx in [0, 1, 2]
    ]

    passages = passage.find_simultaneous_passages(
        dt=spobj_delta_secs,
        space_object=spobj,
        states=spobj_state[:3, ...],
        tx_station=tx_stn,
        rx_stations=rx_stns,
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

    fence_sch = pointing.fence_scanning(
        start_time=start_time_dt64,
        end_time=end_time,
        azimuth=90,  # sweep from east to west
        min_elevation=0,
        pointings_per_cycle=40,
        scan_range=scan_ranges,
        tx_station=tx_stn,
        rx_stations=rx_stns,
        exp_id=exp_detail.id,
        slice_duration=exp_detail.slice_duration,
    )

    spobj_abs_times = spobj_delta_secs * np.timedelta64(1, "s") + start_time_dt64
    spobj_abs_time_indexer: pd.Series[int] = pd.Series(
        np.arange(len(spobj_abs_times)), index=spobj_abs_times
    )
    result_states: list[tx_rx_pair_state.TxRxPairState] = []
    for rx_stn in rx_stns:
        txrx_state = tx_rx_pair_state.gather_from_passages_schedule_dataframe(
            passages=passages,
            sch=fence_sch,
        )[(tx_stn.uid, rx_stn.uid)]

        # TODO: this is slightly cleaner but does not work with multiindex
        # intersection = txrx_state.index.get_level_values("time").intersection(
        #     spobj_abs_time_indexer.index
        # )

        # reindexed_txrx_state = txrx_state.reindex(intersection)
        # reindexed_spobj_state = spobj_state[:, spobj_abs_time_indexer.reindex(intersection)]

        txrx_state_time_idx = txrx_state.index.get_level_values("time")
        txrx_state_mask = txrx_state_time_idx.isin(spobj_abs_time_indexer.index)

        reindexed_txrx_state = txrx_state[txrx_state_mask]
        reindexed_spobj_state = spobj_state[
            :, spobj_abs_time_indexer.reindex(txrx_state_time_idx[txrx_state_mask])
        ]

        result_state = tx_rx_pair_state.simulate(
            txrx_state=reindexed_txrx_state,
            spobj_state=reindexed_spobj_state,
            spobj_diameter=spobj.d,
            spobj_radar_albedo=spobj.properties.get("radar_albedo", 1.0),
            tx_station=tx_stn,
            rx_station=rx_stn,
            exp_detail_map={exp_detail.id: exp_detail},
        )
        result_states.append(result_state)

    # assert the number and info of passages are expected
    assert len(passages) == 1  # 1 `Passage` in total
    assert passages[0].tx_station.uid == tx_stn.uid  # with tx_stn in tx_station
    assert (all(
        rx_station.uid == rx_stn.uid for rx_station, rx_stn in zip(passages[0].rx_stations, rx_stns)
    ))  # with rx_stns in rx_stations; fmt: skip

    for obs in obss_dict[0]:
        rx_schedule_slice = obs.index_into_schedule_dataframe(fence_sch).rx
        simult_num = rx_schedule_slice[_SK.simult_num][0]

        # assert that simult_num is the same over the same observation
        assert (rx_schedule_slice[_SK.simult_num] == simult_num).all()

        # the checks below only make sense for rx station 0
        if obs.passage.rx_stations[0].uid != 0:
            continue

        # assert that we are pointing at scan_ranges
        # NOTE: this is based on the assumption that pointings at same direction but at different scan range
        #   are scheduled in in the same order as `scan_ranges`, and without gaps
        # rx_pointing = rx_schedule_slice[_SK.pointing][:, 0]
        rx_pointing_diff = (
            np.linalg.norm(rx_schedule_slice[_SK.pointing], axis=0) - scan_ranges[simult_num]
        )
        assert (rx_pointing_diff < pointing_range_equality_thld).all()

        # assert the start and end time of the observation is as expected
        # TODO: this can offset pretty large when we have large sampling time interval, is there better way to test it?
        assert abs(
            rx_schedule_slice[_SK.start_time][0] - expected_passage_start_time
        ) < np.timedelta64(int(dsec_sampling_intv), "s")
        assert abs(
            rx_schedule_slice[_SK.end_time][-1] - expected_passage_end_time
        ) < np.timedelta64(int(dsec_sampling_intv), "s")

    # assert that at all rx_pointing from "2nd rx station, 2nd scan range"
    # is the same as those with same `time` but from "1st rx station, 2nd scan range"
    obs_ref = next(
        obs
        for obs in obss_dict[0]
        if obs.passage.tx_station.uid == tx_0_stn.uid
        and obs.passage.rx_stations[0].uid == rx_0_stn.uid
        and obs.index_into_schedule_dataframe(fence_sch).rx[_SK.simult_num][0]
        == 1  # i.e. the 2nd scan range
    )

    obs_subj = next(
        obs
        for obs in obss_dict[0]
        if obs.passage.tx_station.uid == tx_0_stn.uid
        and obs.passage.rx_stations[0].uid == rx_1_stn.uid
        and obs.index_into_schedule_dataframe(fence_sch).rx[_SK.simult_num][0]
        == 1  # i.e. the 2nd scan range
    )

    obs_ref_rx_station = obs_ref.passage.rx_stations[0]
    obs_ref_state_slice = obs_ref.get_state_slice()
    obs_subj_rx_station = obs_subj.passage.rx_stations[0]
    obs_subj_state_slice = obs_subj.get_state_slice()

    obs_subj_pointings_in_ecef = (
        enu_to_ecef(
            lat=obs_subj_rx_station.ecef_lat,
            lon=obs_subj_rx_station.ecef_lon,
            alt=obs_subj_rx_station.ecef_alt,
            enu=obs_subj_state_slice[
                [_SuK.rx_pointing_e, _SuK.rx_pointing_n, _SuK.rx_pointing_u]
            ].T.to_numpy(),
            degrees=True,
        )
        + obs_subj_rx_station.ecef[:, np.newaxis]
    )
    obs_ref_pointings_at_obs_subj_start_times_in_ecef = (
        enu_to_ecef(
            lat=obs_ref_rx_station.ecef_lat,
            lon=obs_ref_rx_station.ecef_lon,
            alt=obs_ref_rx_station.ecef_alt,
            enu=obs_ref_state_slice.loc[obs_subj_state_slice.index][
                [_SuK.rx_pointing_e, _SuK.rx_pointing_n, _SuK.rx_pointing_u]
            ].T.to_numpy(),
            degrees=True,
        )
        + obs_ref_rx_station.ecef[:, np.newaxis]
    )

    assert np.all(
        (obs_subj_pointings_in_ecef - obs_ref_pointings_at_obs_subj_start_times_in_ecef)
        < pointing_range_equality_thld
    )

    return

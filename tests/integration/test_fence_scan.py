"""
We check against an imaginary circular orbit which
- co-rotate with Earth (i.e. a circular orbit on a earth fixed coordinate system)
- at 90deg inclination, 0 deg longitude
- simulation start from the orbit's intersection with earth's equatorial plane, passing south hemisphere then north hemisphere

we use 2 rx stations in this setup:

for rx stations:
- 1st one is the same as tx station, and therefore should have the exact same pointings as tx station
- 2nd one is the same as tx station, except with a small, non-zero min_elevation. It should produce less measurements.
"""

import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time
from astropy.constants import R_earth  # type: ignore
import pyant, pyorb, spacecoords
from sorts.types import Float64_as_sec, Float64_as_deg
from sorts import (
    types,
    utils,
    schedule,
    pointing,
    radar,
    propagator,
    passage,
    SpaceObject,
    simulation,
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
    earth_radius: np.float64 = R_earth.value  # in meters
    spobj_orbital_radius = 7000e3  # in meters
    scan_ranges = np.array([spobj_orbital_radius, spobj_orbital_radius * 1.2], dtype=np.float64)
    num_prop_steps = 60
    min_elevation = 5.0

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

    # introduce non-zero min_elevation to `rx_stns[1]` to test its effect
    rx_stns[1].min_elevation = min_elevation

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
        pairs_ls = [
            schedule.get_tx_rx_pointing_pairs(
                sch=fence_sch,
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

        # TODO: this is slightly cleaner but does not work with multiindex
        # intersection = txrx_state.index.get_level_values("time").intersection(
        #     spobj_abs_time_indexer.index
        # )

        # reindexed_txrx_state = txrx_state.reindex(intersection)
        # reindexed_spobj_state = spobj_state[:, spobj_abs_time_indexer.reindex(intersection)]

        _K = simulation.TxRxPairStateKey

        txrx_state = txrx_state.set_index(_K.time)
        txrx_state_mask = txrx_state.index.isin(spobj_abs_time_indexer.index)

        reindexed_txrx_state = txrx_state[txrx_state_mask]
        reindexed_spobj_state = spobj_state[
            :, spobj_abs_time_indexer.reindex(txrx_state.index[txrx_state_mask])
        ]

        # put time column back to df body
        txrx_state = txrx_state.reset_index()
        reindexed_txrx_state = reindexed_txrx_state.reset_index()

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

    _K = schedule.TxRxPointingPairsKey

    # assert the number and info of passages are expected
    assert len(passages) == 1  # 1 `Passage` in total
    assert passages[0].tx_station.uid == tx_stn.uid  # with tx_stn in tx_station
    assert (all(
        rx_station.uid == rx_stn.uid for rx_station, rx_stn in zip(passages[0].rx_stations, rx_stns)
    ))  # with rx_stns in rx_stations; fmt: skip

    # assert that we are only pointing in the expected scan_ranges
    for result_state in result_states:
        enus = result_state[[_K.rx_pointing_e, _K.rx_pointing_n, _K.rx_pointing_u]].to_numpy().T
        azelrs = spacecoords.spherical.cart_to_sph(enus, degrees=True)

        isclose_mat = np.isclose(scan_ranges, azelrs[2][:, np.newaxis])
        assert np.all(np.any(isclose_mat, axis=1))

    # assert that `result_states[0]` has more values than `result_states[1]`
    # and all missing values are due to space object being
    # outside of the min_elevation of some stations
    assert len(result_states[0]) > len(result_states[1])

    midx_0 = pd.MultiIndex.from_frame(result_states[0][[_K.exp_num, _K.rx_simult_num, _K.time]])
    midx_1 = pd.MultiIndex.from_frame(result_states[1][[_K.exp_num, _K.rx_simult_num, _K.time]])
    midx_diff = midx_0.difference(midx_1)
    missing_rows_mask = midx_0.isin(midx_diff)

    missing_rows_enus = (
        result_states[0][missing_rows_mask][[_K.rx_pointing_e, _K.rx_pointing_n, _K.rx_pointing_u]]
        .to_numpy()
        .T
    )
    missing_rows_azelrs = spacecoords.spherical.cart_to_sph(missing_rows_enus, degrees=True)

    assert np.all(missing_rows_azelrs[1] < min_elevation)

    return

"""
We check against an imaginary circular orbit which
- co-rotate with Earth
- at 90deg inclination, 0 deg longitude
- simulation start from the orbit's intersection with earth's equatorial plane, passing south hemisphere then north hemisphere

we use 2 rx station and 2 scan ranges in this setup:

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
from astropy.time import Time
from astropy.constants import R_earth  # type: ignore
import pyant, pyorb
from sorts import types, schedule, interpolation
from sorts.types import Float64_as_sec, Float64_as_deg, Float_as_sec, Float_as_m
from sorts.utils import to_datetime64_us
from sorts.frames import enu_to_ecef
from sorts.propagator import Kepler, KeplerSettings
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.controller.fence_scan_controller import FenceScanController
from sorts.simulation import stx_mrx_simulation, StxMrxSimulation
from sorts.simulation import InterpolatedPropagation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ref: https://docs.pytest.org/en/stable/how-to/xunit_setup.html#method-and-function-level-setup-teardown
def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


earth_radius: np.float64 = R_earth.value  # in meters
spobj_orbital_radius = 7000e3  # in meters

float_equality_thld = 1e-9
# TODO: re-eval this threshold
pointing_range_equality_thld: Float_as_m = 1e-6

# TODO: re-eval the `control_slice_duration` value, need to be fast but still accurate enough for testing
# control_slice_duration = np.timedelta64(10_000, "us")  # 10ms
control_slice_duration = np.timedelta64(1_000_000, "us")  # 1s

dt_equality_thld = control_slice_duration
dsec_sampling_intv: Float_as_sec = 30
scan_ranges = np.array([10, spobj_orbital_radius], dtype=np.float64)
simu_num = len(scan_ranges)

_SK = schedule.ScheduleKey
_SuK = stx_mrx_simulation.SimulationUnitKey


def south_to_north_circular_orbit_test():
    spobj_orbital_period: Float64_as_sec = pyorb.orbital_period(
        spobj_orbital_radius, pyorb.GM_earth
    )

    passage_angular_duration: Float64_as_deg = np.degrees(
        np.arccos(earth_radius / spobj_orbital_radius) * 2
    )
    passage_duration: Float64_as_sec = passage_angular_duration / 360 * spobj_orbital_period

    start_time = Time("2025-01-01T02:45:00", format="isot", scale="utc")  # simulation start time
    expected_passage_start_time = to_datetime64_us(start_time) + (
        spobj_orbital_period - passage_duration
    ) / 2 * np.timedelta64(int(1e6), "us")
    expected_passage_end_time = expected_passage_start_time + passage_duration * np.timedelta64(
        int(1e6), "us"
    )
    end_time = to_datetime64_us(start_time) + spobj_orbital_period * np.timedelta64(
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

    prop_interp = InterpolatedPropagation.from_space_object(
        space_object=spobj,
        propagator=Kepler(KeplerSettings()),
        interpolator_class=interpolation.Legendre8,
        start_time=start_time,
        end_time=Time(end_time),
        time_step=dsec_sampling_intv,
    )

    tx_0_stn = Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=pyant.models.Isotropic(),
        frequency=233e6,  # same as eisat_3d
        beam_parameters=pyant.models.IsotropicParams(),
        uid=0,
    )

    # NOTE: physically the same as tx_0_stn, but has a different id;
    #   this is done so that we can retain both tx and rx pointings in the schedule,
    #   as FenceScanController will drop the rx entries if both tx and rx use the same station.
    rx_0_stn = Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=pyant.models.Isotropic(),
        frequency=233e6,  # same as eisat_3d
        beam_parameters=pyant.models.IsotropicParams(),
        uid=1,
    )

    # An offseted station for testing behaviours related to `min_elevation`
    rx_1_stn = Station(
        lat=1e-2,
        lon=0.0,
        alt=0.0,
        min_elevation=5.0,
        beam=pyant.models.Isotropic(),
        frequency=233e6,  # same as eisat_3d
        beam_parameters=pyant.models.IsotropicParams(),
        uid=2,
    )

    fence_scan_ctrl = FenceScanController.from_scan_spec(
        tx_station=tx_0_stn,
        rx_stations=[rx_0_stn, rx_1_stn],
        exp_detail=types.ExperimentDetail(
            id=0,
            coh_int_bandwidth=1.0,
            ipp=1.0,
            pulse_length=1.0,
            power=5000000.0,
            bandwidth=52.08333333333333,
            duty_cycle=1.0,
            noise_temp=150.0,
            slice_duration=control_slice_duration,
        ),
        azimuth=90,  # sweep from east to west
        min_elevation=0,
        pointings_per_cycle=40,
        scan_range=scan_ranges,
    )

    fence_sch = fence_scan_ctrl.generate(start_time, end_time)

    sim = StxMrxSimulation.from_controllers(
        controllers=[fence_scan_ctrl],
        schedule=fence_sch,
        epoch=start_time,
        start_time=start_time,
        end_time=end_time,
        space_objects=[spobj],
        interpolated_propagations=[prop_interp],
    )

    obss_dict, sim_units_dict = sim.run()

    # assert the number of `Passage`, `SimulationUnit` and `Observation` are expected
    assert sum(len(sim_unit_list) for sim_unit_list in sim_units_dict.values()) == 2 # 2 `SimulationUnit` in total; fmt: skip
    assert len(sim_units_dict[0][0].passages) == 1  # each unit has 1 `Passage`
    assert len(sim_units_dict[0][1].passages) == 1  # each unit has 1 `Passage`
    assert sum(len(obss_list) for obss_list in obss_dict.values()) == 3 # 3 `Observaition` in total; fmt: skip

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

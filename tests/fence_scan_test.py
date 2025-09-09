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
- 2nd one is slightly offseted, with a small, non-zero min_elevation. It should produce 2 observations:
  - 1st observation should be an empty observation for corresponding to the 1st scan range
  - 2nd observation should points to the same location as the 2nd observation of the 1st station, with a bit of masking due to `min_elevation`
"""

import logging
import numpy as np
from astropy.time import Time
from astropy.constants import R_earth  # type: ignore
from pyant import Beam
import pyorb
from sorts.types import Float64_as_sec, Float64_as_deg, Float_as_sec, Float_as_m
from sorts.utils import to_datetime64_us
from sorts.frames import enu_to_ecef
from sorts.interpolation import Legendre8
from sorts.propagator import Kepler
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.schedule import Schedule
from sorts.controller.fence_scan_controller import FenceScanController
from sorts.simulation import stx_mrx_simulation, StxMrxSimulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ref: https://docs.pytest.org/en/stable/how-to/xunit_setup.html#method-and-function-level-setup-teardown
def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


float_equality_thld = 1e-9
# TODO: re-eval this threshold
pointing_range_equality_thld: Float_as_m = 1e-6

# TODO: re-eval the `control_slice_duration` value, need to be fast but still accurate enough for testing
# control_slice_duration = np.timedelta64(10_000, "us")  # 10ms
control_slice_duration = np.timedelta64(1_000_000, "us")  # 1s

dt_equality_thld = control_slice_duration
dsec_sampling_intv: Float_as_sec = 30
scan_ranges = np.array([10, 7e6], dtype=np.float64)
simu_num = len(scan_ranges)

_SK = Schedule._K
_SuK = stx_mrx_simulation.simulation_unit._K


def south_to_north_circular_orbit_test():
    earth_radius: np.float64 = R_earth.value  # in meters
    spobj_orbital_radius = 7000e3  # in meters
    spobj_orbital_period: Float64_as_sec = pyorb.orbital_period(
        spobj_orbital_radius, pyorb.GM_earth
    )

    passage_angular_duration: Float64_as_deg = np.degrees(
        np.arccos(earth_radius / spobj_orbital_radius) * 2
    )
    passage_duration: Float64_as_sec = passage_angular_duration / 360 * spobj_orbital_period

    start_time = Time("2025-01-01 02:45:00")  # simulation start time
    expected_passage_start_time = to_datetime64_us(start_time) + (
        spobj_orbital_period - passage_duration
    ) / 2 * np.timedelta64(int(1e6), "us")
    expected_passage_end_time = expected_passage_start_time + passage_duration * np.timedelta64(
        int(1e6), "us"
    )
    end_time = to_datetime64_us(start_time) + spobj_orbital_period * np.timedelta64(
        int(1e6), "us"
    )  # simulation end time

    spobj = SpaceObject(
        oid=0,
        propagator=Kepler,
        propagator_options={"settings": {"in_frame": "GCRS", "out_frame": "GCRS"}},
        a=spobj_orbital_radius,
        e=0,
        i=90,
        raan=0,
        aop=0,
        mu0=180,
        epoch=start_time,
        parameters={"d": 1.0},  # diameter of the spobj
    )

    # TODO: correct the return type of the `Beam.gain` base class method
    class IsotropicBeam(Beam):
        def gain(self, k, ind=None, polarization=None, **kwargs):
            if len(k.shape) == 1:
                return 1.0
            elif len(k.shape) == 2:
                return np.full(k.shape[1], 1.0, dtype=np.float64)
            else:
                raise RuntimeError(f"unexpected shape of k: {k.shape}")

    tx_rx_0_stn = Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=IsotropicBeam(
            azimuth=0.0,
            elevation=0.0,
            frequency=233e6,  # same as eisat_3d
        ),
        uid="test_station, tx-rx, 0",
    )

    # An offseted station for testing behaviours related to `min_elevation`
    rx_1_stn = Station(
        lat=1e-2,
        lon=0.0,
        alt=0.0,
        min_elevation=5.0,
        beam=IsotropicBeam(
            azimuth=0.0,
            elevation=0.0,
            frequency=233e6,  # same as eisat_3d
        ),
        uid="test_station, rx, 1",
    )

    def dsec_sampler(orbit, start_time, end_time):
        return np.arange(0, (end_time - start_time) / np.timedelta64(1, "s"), 30, dtype=np.float64)

    fence_scan_ctrl = FenceScanController.from_scan_spec(
        tx_station=tx_rx_0_stn,
        rx_stations=[tx_rx_0_stn, rx_1_stn],
        exp_detail={
            "id": 0,
            "coh_int_bandwidth": 1.0,
            "ipp": 1.0,
            "pulse_length": 1.0,
            "power": 5000000.0,
            "bandwidth": 52.08333333333333,
            "duty_cycle": 1.0,
            "noise_temp": 150.0,
            "slice_duration": control_slice_duration,
        },
        azimuth=90,  # sweep from east to west
        min_elevation=0,
        pointings_per_cycle=40,
        scan_range=scan_ranges,
    )

    fence_schs = fence_scan_ctrl.generate(start_time, end_time)

    exp_detail_map = {fence_scan_ctrl.spec["exp_detail"]["id"]: fence_scan_ctrl.spec["exp_detail"]}

    sim = StxMrxSimulation.from_spec(
        {
            "tx_station": tx_rx_0_stn,
            "tx_schedule": fence_schs.tx_schedule,
            "rx_stations": [tx_rx_0_stn, rx_1_stn],
            "rx_schedules": fence_schs.rx_schedules,
            "exp_detail_map": exp_detail_map,
            "epoch": start_time,
            "start_time": start_time,
            "end_time": end_time,
            "space_objects": [spobj],
            "dsec_sampler": dsec_sampler,
            "interpolator_class": Legendre8,
        }
    )

    obss, sim_units = sim.run()

    # assert the number of `Passage`, `SimulationUnit` and `Observation` are expected
    assert len(sim_units) == 2
    assert len(sim_units[0].passages) == 1
    assert len(sim_units[1].passages) == 1
    assert len(obss) == simu_num * 2

    for obs, scan_range in zip(obss[0:3], scan_ranges):
        # assert that we are pointing at scan_ranges
        # NOTE: this is based on the assumption that pointings at same direction but at different scan range
        #   are scheduled in in the same order as `scan_ranges`, and without gaps
        rx_pointing = obs.get_schedule_slice().rx._data[_SK.pointing][:, 0]
        assert (np.linalg.norm(rx_pointing) - scan_range) < pointing_range_equality_thld

        # assert the start and end time of the observation is as expected
        # TODO: this can offset pretty large when we have large sampling time interval, is there better way to test it?
        assert abs(
            obs.get_schedule_slice().rx._data[_SK.start_time][0] - expected_passage_start_time
        ) < np.timedelta64(int(dsec_sampling_intv), "s")
        assert abs(
            obs.get_schedule_slice().rx._data[_SK.end_time][-1] - expected_passage_end_time
        ) < np.timedelta64(int(dsec_sampling_intv), "s")

    # assert that obss[2] is empty
    assert len(obss[2].get_state_slice()[_SuK.time]) == 0

    # assert that at all rx_pointing of `obss[3]` in ecef is the same as those with same `time` in `obss[1]`
    obs_1_rx_station = obss[1].passage["rx_station"]
    obs_1_state_slice = obss[1].get_state_slice()
    obs_3_rx_station = obss[3].passage["rx_station"]
    obs_3_state_slice = obss[3].get_state_slice()

    obs_3_pointings_in_ecef = (
        enu_to_ecef(
            lat=obs_3_rx_station.ecef_lat,
            lon=obs_3_rx_station.ecef_lon,
            alt=obs_3_rx_station.ecef_alt,
            enu=obs_3_state_slice[_SuK.rx_pointing],
            degrees=True,
        )
        + obs_3_rx_station.ecef[:, np.newaxis]
    )
    obs_0_pointings_at_obs_3_start_times_in_ecef = (
        enu_to_ecef(
            lat=obs_1_rx_station.ecef_lat,
            lon=obs_1_rx_station.ecef_lon,
            alt=obs_1_rx_station.ecef_alt,
            enu=obs_1_state_slice.loc[{_SuK.time: obs_3_state_slice[_SuK.time]}][_SuK.rx_pointing],
            degrees=True,
        )
        + obs_1_rx_station.ecef[:, np.newaxis]
    )

    assert np.all(
        (obs_3_pointings_in_ecef - obs_0_pointings_at_obs_3_start_times_in_ecef)
        < pointing_range_equality_thld
    )

    return

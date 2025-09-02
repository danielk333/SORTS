"""
We check against an imaginary circular orbit which
- co-rotate with Earth
- at 90deg inclination, 0 deg longitude
- simulation start from the orbit's intersection with earth's equatorial plane, passing south hemisphere then north hemisphere
"""

# TODO: should add more description/explanation of the setup

import logging
import numpy as np
from astropy.time import Time
from astropy.constants import R_earth  # type: ignore
from pyant import Beam
import pyorb
from sorts.types import Float64_as_sec, Float64_as_deg, Float_as_sec, Float_as_deg, Float_as_m
from sorts.utils import to_datetime64_us
from sorts.interpolation import Legendre8
from sorts.propagator import Kepler
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.controller_v2.fence_scan_controller import FenceScanController
from sorts.simulation_v2 import StxMrxSimulation, SimulationUnit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ref: https://docs.pytest.org/en/stable/how-to/xunit_setup.html#method-and-function-level-setup-teardown
def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


float_equality_thld = 1e-9
# NOTE: this is much more lenient than `float_equality_thld`, because pointing calc involves trigs and other less precise funcs
# TODO: use ENU for pointings in schedule? it allows `pointing_equality_thld = 1e-3`
pointing_equality_thld: Float_as_deg = 5e-3
pointing_equality_thld_loose: Float_as_deg = 1  # even 0.5 deg fails
# TODO: re-eval this threshold after converting schedule data to use ENU instead of Azelr
pointing_range_equality_thld: Float_as_m = 5.0

# TODO: re-eval the `control_slice_duration` value, need to be fast but still accurate enough for testing
# control_slice_duration = np.timedelta64(10_000, "us")  # 10ms
control_slice_duration = np.timedelta64(1_000_000, "us")  # 1s

dt_equality_thld = control_slice_duration
dsec_sampling_intv: Float_as_sec = 30
simu_num = 3
scan_ranges = np.linspace(300e3, 1000e3, num=simu_num, dtype=np.float64)

_SuK = SimulationUnit._K


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

    class IsotropicBeam(Beam):
        def gain(self, k, ind=None, polarization=None, **kwargs):
            if len(k.shape) == 1:
                return 1.0
            elif len(k.shape) == 2:
                return np.full(k.shape[1], 1.0, dtype=np.float64)
            else:
                raise RuntimeError(f"unexpected shape of k: {k.shape}")

    tx_stn = Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=IsotropicBeam(
            azimuth=0.0,
            elevation=0.0,
            frequency=233e6,  # same as eisat_3d
        ),
        uid="test_station, tx, 0",
    )

    rx_stn = Station(
        lat=1e-2,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=IsotropicBeam(
            azimuth=0.0,
            elevation=0.0,
            frequency=233e6,  # same as eisat_3d
        ),
        uid="test_station, rx, 0",
    )

    def dsec_sampler(orbit, start_time, end_time):
        return np.arange(0, (end_time - start_time) / np.timedelta64(1, "s"), 30, dtype=np.float64)

    fence_scan_ctrl = FenceScanController.from_scan_spec(
        tx_station=tx_stn,
        rx_stations=[rx_stn],
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
        min_elevation=70,
        pointings_per_cycle=40,
        scan_range=scan_ranges,
    )

    fence_schs = fence_scan_ctrl.generate(start_time, end_time)

    exp_detail_map = {fence_scan_ctrl.spec["exp_detail"]["id"]: fence_scan_ctrl.spec["exp_detail"]}

    sim = StxMrxSimulation.from_spec(
        {
            "tx_station": tx_stn,
            "tx_schedule": fence_schs.tx_schedule,
            "rx_stations": [rx_stn],
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

    sim_units, passage_obs_idxers_pairs = sim.run()
    sim_unit = sim_units[0]

    # assert there is 1 passage and `simu_num` number of observations
    assert len(passage_obs_idxers_pairs) == 1
    assert len(passage_obs_idxers_pairs[0][1]) == simu_num

    # TODO: simplify the indexer/simulation output so indexing into the observation is less cryptic
    # assert we are getting observations with the intended scan ranges
    for i, scan_range in enumerate(scan_ranges):
        assert (
            (
                fence_schs.rx_schedules[0]
                .filter_by_time_range(passage_obs_idxers_pairs[0][0]["time_range"])
                ._data.loc[{"start_time": passage_obs_idxers_pairs[0][1][i]["rx_indexer"]}][
                    "pointing"
                ]
                .loc["r"][0]
            )
            - scan_range
        ) < pointing_range_equality_thld

    for sim_unit in sim_units:
        # TODO: assert `El` componend of tx pointings swing between 0 and 90? or we can check it in pointing generation unit test instead
        # assert `Az` componend of tx pointings is 90 or 270
        assert np.all(
            abs(
                np.sort(np.unique(sim_unit._state_data[_SuK.tx_pointing][0]))
                - np.array([90, 270], dtype=np.float64)
            )
            < float_equality_thld
        )

        # assert the start and end time of the observation is as expected
        # TODO: this can offset pretty large when we have large sampling time interval, is there better way to test it?
        assert abs(
            sim_unit._state_data[_SuK.time].min().to_numpy() - expected_passage_start_time
        ) < np.timedelta64(int(dsec_sampling_intv), "s")
        assert abs(
            sim_unit._state_data[_SuK.time].max().to_numpy() - expected_passage_end_time
        ) < np.timedelta64(int(dsec_sampling_intv), "s")

    return

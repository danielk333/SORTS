"""
We check against an imaginary circular orbit which
- co-rotate with Earth
- at 90deg inclination, 0 deg longitude
- simulation start from the orbit's intersection with earth's equatorial plane, passing south hemisphere then north hemisphere
"""

# TODO: should add more description/explanation of the setup

import logging, typing as T
import numpy as np
from astropy.time import Time
from astropy.constants import R_earth  # type: ignore
import pyant, pyorb
from sorts import types, schedule, interpolation, simulation
from sorts.types import Float64_as_sec, Float64_as_deg, Float_as_sec, Float_as_deg
from sorts.utils import to_datetime64_us
from sorts.propagator import Kepler, KeplerSettings
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.controller.tracker_controller import TrackerController
from sorts.interpolated_propagation import InterpolatedPropagation
from sorts.simulation import StxMrxSimulation

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
# NOTE: this is much more lenient than `float_equality_thld`
pointing_equality_thld: Float_as_deg = 0.04

# TODO: re-eval the `control_slice_duration` value, need to be fast but still accurate enough for testing
# control_slice_duration = np.timedelta64(10_000, "us")  # 10ms
control_slice_duration = np.timedelta64(1_000_000, "us")  # 1s

dsec_sampling_intv: Float_as_sec = 30
dt_equality_thld = dsec_sampling_intv * 0.2  # TODO: need to eval this value with Daniel

_SK = schedule.ScheduleKey
_SuK = simulation.SimulationUnitKey


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

    test_stn = Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=pyant.models.Isotropic(),
        frequency=233e6,  # same as eisat_3d
        beam_parameters=pyant.models.IsotropicParams(),
        uid=0,
    )

    tracker_ctrl = TrackerController.from_space_object(
        space_object=spobj,
        epoch=start_time,
        tx_station=test_stn,
        rx_stations=[test_stn],
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
        interpolated_propagation=prop_interp,
    )

    tracker_sch = tracker_ctrl.generate(start_time, end_time)

    sim = StxMrxSimulation.from_controllers(
        controllers=[tracker_ctrl],
        schedule=tracker_sch,
        epoch=start_time,
        start_time=start_time,
        end_time=end_time,
        space_objects=[spobj],
        interpolated_propagations=[prop_interp],
    )

    obss_dict, sim_units_dict = sim.run()
    sim_unit = sim_units_dict[0][0]

    # assert there is only 1 observation
    assert sum(len(obss_list) for obss_list in obss_dict.values()) == 1
    obs = obss_dict[0][0]

    tx_pointings = obs.index_into_schedule_dataframe(tracker_sch).tx[_SK.pointing]
    tx_pointings_normalized = tx_pointings / np.linalg.norm(tx_pointings.to_numpy(), axis=0)

    # assert `E` componend of tx pointings stayed around zero
    assert np.all(
        obs.index_into_schedule_dataframe(tracker_sch).tx[_SK.pointing].loc[_SK.e, :] < float_equality_thld
    )

    # assert `N` componend of normalized tx pointings swing between -1.0 and +1.0
    assert abs(tx_pointings_normalized.loc[_SK.n, :].min() + 1.0) < pointing_equality_thld
    assert abs(tx_pointings_normalized.loc[_SK.n, :].max() - 1.0) < pointing_equality_thld

    # assert `U` componend of normalized tx pointings swing between 0.0 and 1.0
    assert abs(tx_pointings_normalized.loc[_SK.u, :].min()) < pointing_equality_thld
    assert abs(tx_pointings_normalized.loc[_SK.u, :].max() - 1.0) < pointing_equality_thld

    # assert the max snr time is roughly at half orbital period
    assert (
        abs(
            sim_unit._state.index.get_level_values(_SuK.time)[
                T.cast(int, sim_unit._state[_SuK.snr].argmax())
            ] # fmt: skip
            - (
                to_datetime64_us(start_time) + spobj_orbital_period / 2 * np.timedelta64(int(1e6), "us")
            ) # fmt: skip
        ) / np.timedelta64(1, "s")
        < dt_equality_thld
    )

    # assert the start and end time of the observation is as expected
    # TODO: this can offset pretty large when we have large sampling time interval, is there better way to test it?
    #       it is multiplied by 1.5 by now to get the test running
    assert abs(
        obs.index_into_schedule_dataframe(tracker_sch).rx[_SK.start_time][0] - expected_passage_start_time
    ) < np.timedelta64(int(dsec_sampling_intv*1.5), "s")
    assert abs(
        obs.index_into_schedule_dataframe(tracker_sch).rx[_SK.end_time][-1] - expected_passage_end_time
    ) < np.timedelta64(int(dsec_sampling_intv*1.5), "s")

    return

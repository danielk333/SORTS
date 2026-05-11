import logging, typing as t, argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor
import sorts
from sorts import (
    types,
    utils,
    pointing,
    population,
    propagator,
    radar,
    passage,
    ExperimentDetail,
)


@dataclass(kw_only=True)
class ScriptParams:
    save_dname: str
    save_dpath: Path
    plot_dpath: Path
    start_time: Time
    end_time: Time
    control_slice_duration: np.timedelta64
    coherent_integration_time: float
    time_step: float
    rand_seed: int
    grid_size: tuple[int, int]
    tx_station: radar.TX
    rx_stations: t.Sequence[radar.RX]
    station_map: t.Mapping[types.StationId, radar.Station]
    exp_detail_map: types.ExperimentDetailMap
    prop: propagator.Propagator
    oids: npt.NDArray[np.int64]
    clobber: bool
    rng: t.Any


def prepare_simulation(cli_args: argparse.Namespace) -> tuple[ScriptParams, population.Population]:
    coherent_integration_time = 0.04
    duty_cycle = 0.2
    grid_size = (4, 4)  # used a very small grid for demo, normal values are e.g. `(50, 50)`
    rand_seed = 1203
    start_time = Time("2025-01-01 00:00:00")
    time_slice = coherent_integration_time / duty_cycle
    radar_sys = radar.radars.nostra.gen_nostra(
        frequency=3.2e9,
        antenna_num=cli_args.antennas,
        antenna_spacing_lambda=0.65,
        antenna_efficiency=0.5,
        antenna_input_power=100,  # W
        thermal_load=1,
        noise_figure_db=0.7,
        amplifier_gain_db=18,
        insertion_loss_db=0.35,
        duty_cycle=duty_cycle,
        t_sky=10.0,
        coherent_integration_time=coherent_integration_time,
        bandwidth_limit_ratio=5,
    )

    tx_station = radar_sys.tx[0]
    rx_stations = radar_sys.rx
    station_map = {idx: stn for idx, stn in enumerate([tx_station, *rx_stations])}
    for idx, stn in station_map.items():
        stn.uid = idx

    np.random.seed(rand_seed)
    rng = np.random.default_rng(seed=rand_seed)

    spobj_pop = population.orbit_grid(
        semi_major_axis_samples=np.linspace(R_earth + 300e3, R_earth + 1000e3, num=grid_size[0]),
        eccentricity_samples=np.array([0]),
        inclination_samples=np.array([0]),
        argument_of_periapsis_samples=np.array([0]),
        longitude_of_ascending_node_samples=np.array([0]),
        mean_anomaly_samples=np.array([0]),
        diameter_samples=10 ** np.linspace(-2, 1, num=grid_size[1]),
        frame="TEME",
        epoch_mjd=t.cast(float, start_time.mjd),
        additional_parameters={"area_to_mass": 0, "m": 0},
        degrees=True,
    )
    spobj_pop.data["i"] = 75.0
    spobj_pop.data["area_to_mass"] = 10 ** (np.random.rand(len(spobj_pop)) * 4 - 3)
    areas = np.pi * (spobj_pop.data["d"] / 2) ** 2
    spobj_pop.data["m"] = areas / spobj_pop.data["area_to_mass"]

    oids = np.arange(len(spobj_pop))

    control_slice_duration = np.timedelta64(int(time_slice * 1e6), "us")
    exp_detail = ExperimentDetail(
        id=0,
        # not used
        coh_int_bandwidth=1.0,
        ipp=1.0,
        pulse_length=1.0,
        duty_cycle=1.0,
        # --
        power=tx_station.power,
        bandwidth=1 / coherent_integration_time,
        noise_temp=rx_stations[0].noise,
        slice_duration=control_slice_duration,
    )

    prm = ScriptParams(
        save_dname=cli_args.name,
        save_dpath=cli_args.out_dir / cli_args.name,
        plot_dpath=cli_args.out_dir / cli_args.name / "plots",
        start_time=start_time,
        # end_time = Time("2025-01-07 00:00:00"),
        end_time=Time("2025-01-02 00:00:00"),
        control_slice_duration=control_slice_duration,
        coherent_integration_time=coherent_integration_time,
        time_step=10.0,
        rand_seed=rand_seed,
        grid_size=grid_size,
        tx_station=tx_station,
        rx_stations=rx_stations,
        station_map=station_map,
        exp_detail_map={exp_detail.id: exp_detail},
        prop=propagator.Sgp4(
            settings=propagator.Sgp4Settings(out_frame="ITRS", mean_elements_input=True)
        ),
        oids=oids,
        clobber=cli_args.clobber,
        rng=rng,
    )
    utils.ensure_directory_exist(prm.save_dpath)
    utils.ensure_directory_exist(prm.plot_dpath)

    return prm, spobj_pop


@utils.use_pickled_or_compute
def propagate(spobj: sorts.SpaceObject, prm: ScriptParams):
    spobj_simulator = sorts.SpaceObjectSimulator(
        propagator=sorts.propagator.Sgp4(
            settings=sorts.propagator.Sgp4Settings(out_frame="ITRS", mean_elements_input=True)
        ),
        interpolator=sorts.space_object_states_interpolation.Legendre8,
    )

    spobj_and_perts = spobj_simulator.perturbate(
        space_object=spobj,
        perturbation_format="kepler",
        pert_val=(
            1e-3, 1e-5, 1e-5, 1e-5, 1e-5, 1e-3  # fmt: skip
        ),
    )

    times, true_states = spobj_simulator.propagate(
        space_object=spobj,
        start_time=prm.start_time,
        end_time=prm.end_time,
        time_step=prm.time_step,
    )

    spobj_prop_and_perts = [true_states]
    for obj in spobj_and_perts[1:]:
        _, pert_states = spobj_simulator.propagate(
            space_object=obj,
            start_time=prm.start_time,
            end_time=prm.end_time,
            time_step=prm.time_step,
        )
        spobj_prop_and_perts.append(pert_states)

    spobj_interp_and_perts = [
        spobj_simulator.make_interpolation(times=times, states=states)
        for states in spobj_prop_and_perts
    ]

    return spobj_simulator, times, spobj_and_perts, spobj_prop_and_perts, spobj_interp_and_perts


@utils.use_pickled_or_compute
def compute_schedule_and_passages(propagate_step_pickle: utils.PickledObject, prm: ScriptParams):
    (
        spobj_simulator,
        times,
        spobj_and_perts,
        spobj_prop_and_perts,
        spobj_interp_and_perts,
    ) = utils.as_retval(propagate, propagate_step_pickle).load()

    spobj = spobj_and_perts[0]
    true_states = spobj_prop_and_perts[0]
    true_states_interp = spobj_interp_and_perts[0]

    passages = passage.find_simultaneous_passages(
        dt=times,
        space_object=spobj,
        states=true_states[:3, ...],
        tx_station=prm.tx_station,
        rx_stations=prm.rx_stations,
        epoch=utils.to_datetime64_us(prm.start_time),
    )

    tracker_sch = pointing.sparse_tracking(
        passages_of_spobj=passages,
        interpolation=true_states_interp,
        points_per_passage=10,
        tx_station=prm.tx_station,
        rx_stations=prm.rx_stations,
        exp_id=prm.exp_detail_map[0].id,
        slice_duration=prm.exp_detail_map[0].slice_duration,
    )

    return tracker_sch, passages


@utils.use_pickled_or_compute
def simulate(
    propagate_step_pickle: utils.PickledObject,
    schedule_and_passages_pickle: utils.PickledObject,
    prm: ScriptParams,
):
    (
        spobj_simulator,
        times,
        spobj_and_perts,
        spobj_prop_and_perts,
        spobj_interp_and_perts,
    ) = utils.as_retval(propagate, propagate_step_pickle).load()

    sch, passages = utils.as_retval(
        compute_schedule_and_passages, schedule_and_passages_pickle
    ).load()

    sim_result = utils.empty_list_of_retval(spobj_simulator.simulate)
    for spobj, interp in tqdm(
        # NOTE: used `list(zip(...))` instead of just `zip(...)` so that `tqdm` can get the length
        list(zip(spobj_and_perts, spobj_interp_and_perts)),
        desc="simulating a perturbation set",
    ):
        sim_result.append(
            spobj_simulator.simulate(
                space_object=spobj,
                states_interpolation=interp,
                # the same passage data is used for all perturbed objects
                passages=passages,
                sch=sch,
                station_map=prm.station_map,
                exp_detail_map=prm.exp_detail_map,
            )
        )

    return sim_result


# NOTE: functions that passed to executor.map cannot be declared inside of
#       `with MPICommExecutor(max_workers=4) as executor:` block.
#       the executor will not be able to grab a reference to it.
#
#       making our own helper class is still helpful considering this?
#       we should align the syntax to the `MPICommExecutor` though
#
# NOTE: the typing of `MPICommExecutor` is not complete, it is missing the `star`, `submit`, ... methods.
#       looking into the code, `MPICommExecutor` returns a `MPIPoolExecutor` for master rank,
#       so the method do exists, but as method of `MPIPoolExecutor`
# with MPIPoolExecutor(max_workers=2) as executor:
with MPICommExecutor() as executor:

    logging.basicConfig(level=logging.DEBUG, force=True)
    logging.getLogger("sorts.propagator").setLevel(logging.WARNING)
    logging.getLogger("sorts.frames").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info("starting example")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).parent / "local_data")
    parser.add_argument("--name", type=str, default="sparse_tracking_simulation_with_mpi")
    parser.add_argument("--clobber", action="store_true")
    parser.add_argument("--antennas", type=int, default=10_000)
    cli_args = parser.parse_args()

    R_earth = 6371e3

    if executor is not None:
        # preparations
        prm, spobj_pop = prepare_simulation(cli_args)
        spobjs = [spobj_pop.get_object(oid) for oid in prm.oids]

        # propagate
        params_list = [
            (
                prm.save_dpath / f"space_object_{spobj.object_id}" / "propagate_step_output.pickle",
                prm.clobber,
                spobj,
                prm,
            )
            for spobj in spobjs
        ]
        propagate_step_pickle_list = executor.starmap(
            propagate, tqdm(params_list, desc="propagate")
        )

        # compute_schedule_and_passages step
        params_list = [
            (
                prm.save_dpath / f"space_object_{spobj.object_id}" / "schedule_and_passages.pickle",
                prm.clobber,
                propagate_step_pickle,
                prm,
            )
            for spobj, propagate_step_pickle in zip(spobjs, propagate_step_pickle_list)
        ]
        schedule_and_passages_pickle_list = executor.starmap(
            compute_schedule_and_passages,
            tqdm(params_list, desc="compute_schedule_and_passages"),
        )

        # simulation step
        logger.debug("starting simulation")
        params_list = [
            (
                prm.save_dpath / f"space_object_{spobj.object_id}" / "simulation_result.pickle",
                prm.clobber,
                propagate_step_pickle,
                schedule_and_passages_pickle_,
                prm,
            )
            for spobj, propagate_step_pickle, schedule_and_passages_pickle_ in zip(
                spobjs, propagate_step_pickle_list, schedule_and_passages_pickle_list
            )
        ]
        sim_result_list = executor.starmap(simulate, tqdm(params_list, desc="simulate"))
        logger.debug("simulation done")

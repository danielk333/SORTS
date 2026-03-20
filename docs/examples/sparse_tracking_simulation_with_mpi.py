import logging, typing as t, pickle, argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from tqdm import tqdm
from sorts import (
    types,
    utils,
    space_object,
    controller,
    interpolation,
    population,
    propagator,
    radar,
    schedule,
    passage,
    perturbation,
    ExperimentDetail,
    MpiQueuedExecution,
)
from sorts.simulation import stx_mrx_simulation


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("sorts.propagator").setLevel(logging.WARNING)
logging.getLogger("sorts.frames").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info("starting example")

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=Path, default=Path(__file__).parent / "local_data")
parser.add_argument("--name", type=str, default="sparse_tracking_simulation_with_mpi")
parser.add_argument("--clobber", action="store_true")
parser.add_argument("--antennas", type=int, default=10_000)
args = parser.parse_args()

R_earth = 6371e3


@dataclass(kw_only=True)
class SimulationParams:
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


def prepare_simulation(args) -> tuple[SimulationParams, population.Population]:
    ##
    # prepare simulation environment
    ##

    coherent_integration_time = 0.04
    duty_cycle = 0.2
    grid_size = (4, 4)  # used a very small grid for demo, normal values are e.g. `(50, 50)`
    rand_seed = 1203
    start_time = Time("2025-01-01 00:00:00")
    time_slice = coherent_integration_time / duty_cycle
    radar_sys = radar.radars.nostra.gen_nostra(
        frequency=3.2e9,
        antenna_num=args.antennas,
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

    prm = SimulationParams(
        save_dname=args.name,
        save_dpath=args.out_dir / args.name,
        plot_dpath=args.out_dir / args.name / "plots",
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
        clobber=args.clobber,
        rng=rng,
    )
    utils.ensure_directory_exist(prm.save_dpath)
    utils.ensure_directory_exist(prm.plot_dpath)

    return prm, spobj_pop


class Propagate(MpiQueuedExecution[tuple[space_object.SpaceObject, SimulationParams]]):
    def master_process(self):
        prm, spobj_pop = prepare_simulation(args)

        spobjs = [spobj_pop.get_object(oid) for oid in prm.oids]
        worker_job_params = [(spobj, prm) for spobj in spobjs]
        self.mpi_master_proc_loop(worker_job_params)

    def worker_process(self, worker_job_params):
        spobj, prm = worker_job_params
        obj_pth = prm.save_dpath / f"space_object_{spobj.object_id}"
        utils.ensure_directory_exist(obj_pth)

        pert_pth = obj_pth / "pert_obj_propagation_interpolation.pickle"
        if prm.clobber or not pert_pth.exists():
            perturbed_object_groups = perturbation.duplicate_and_perturbate_space_object(
                space_object=spobj,
                propagator=prm.prop,
                interpolator_class=interpolation.Legendre8,
                start_time=prm.start_time,
                end_time=prm.end_time,
                time_step=prm.time_step,
                perturbation_format="kepler",
                pert_val=(
                    1e-3, 1e-5, 1e-5, 1e-5, 1e-5, 1e-3  # fmt: skip
                ),
            )
            utils.safe_pickle(perturbed_object_groups, pert_pth)

            spobj_pth = obj_pth / "spboj_data.pickle"
            prop_interp_pth = obj_pth / "propagation_interpolation.pickle"
            true_spobj, true_prop = perturbed_object_groups[0]
            if prm.clobber or not spobj_pth.exists():
                utils.safe_pickle(true_spobj, spobj_pth)
            if prm.clobber or not prop_interp_pth.exists():
                utils.safe_pickle(true_prop, prop_interp_pth)


class SimulateObs(MpiQueuedExecution[tuple[types.SpaceObjectId, SimulationParams]]):
    def master_process(self):
        prm, spobj_pop = prepare_simulation(args)

        spobjs = [spobj_pop.get_object(oid) for oid in prm.oids]
        worker_job_params = [(spobj.object_id, prm) for spobj in spobjs]
        self.mpi_master_proc_loop(worker_job_params)

    def worker_process(self, worker_job_params):
        object_id, prm = worker_job_params
        obj_pth = prm.save_dpath / f"space_object_{object_id}"
        pert_pth = obj_pth / "pert_obj_propagation_interpolation.pickle"
        with open(pert_pth, "rb") as fh:
            perturbed_object_groups: list[perturbation.SpaceObjectInterpolatedPropagationPair] = (
                pickle.load(fh)
            )

        spobj, prop_interp = perturbed_object_groups[0]
        spobjs = [tup[0] for tup in perturbed_object_groups]
        prop_interps = [tup[1] for tup in perturbed_object_groups]

        sim_pth = obj_pth / "simulation.pickle"
        if prm.clobber or not sim_pth.exists():
            passages = passage.find_simultaneous_passages(
                dt=(
                    (prop_interp.times - utils.to_datetime64_us(prm.start_time))
                    / np.timedelta64(1, "s")
                ),
                space_object=spobj,
                states=prop_interp.states[:3, ...],
                tx_station=prm.tx_station,
                rx_stations=prm.rx_stations,
                epoch=utils.to_datetime64_us(prm.start_time),
            )

            tracker_ctrl = controller.SparseTrackerController.from_space_object(
                tx_station=prm.tx_station,
                rx_stations=prm.rx_stations,
                exp_detail=prm.exp_detail_map[0],
                space_object=spobj,
                epoch=prm.start_time,
                points_per_passage=10,
                interpolator=prop_interp.interpolator,
            )

            tracker_sch = tracker_ctrl.generate(passages)
            schedule_db = schedule.ScheduleDb.from_schedule_dataframes(
                [tracker_sch], ["tracker_sch"], obj_pth / "schedule.sqlite"
            )
            schedule_db.schedule_by_priority()

            sim = stx_mrx_simulation.StxMrxSimulation(
                schedule_db=schedule_db,
                space_objects=spobjs,
                interpolated_propagations=prop_interps,
                passages=passages,
            )
            utils.safe_pickle(sim, sim_pth)
        else:
            with open(sim_pth, "rb") as fh:
                sim: stx_mrx_simulation.StxMrxSimulation = pickle.load(fh)

        obs_pth = obj_pth / "simulation_result.pickle"
        if prm.clobber or not obs_pth.exists():
            logger.debug("starting simulation")

            sim_result = [
                stx_mrx_simulation.simulate(
                    space_object=spobj,
                    interpolated_propagation=interp_prop,
                    # the same passage data is used for all perturbed objects
                    passages=sim.passages,
                    schedule_db=sim.schedule_db,
                    station_map=prm.station_map,
                    exp_detail_map=prm.exp_detail_map,
                )
                for spobj, interp_prop in tqdm(
                    # NOTE: used `list(zip(...))` instead of just `zip(...)` so that `tqdm` can get the length
                    list(zip(sim.space_objects, sim.interpolated_propagations)),
                    desc="simulating",
                )
            ]

            logger.debug("simulation done")

            utils.safe_pickle(sim_result, obs_pth)


try:
    from mpi4py import MPI

    pool_size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    pool_size = 1
    rank = 0

Propagate(
    progress=True,
    is_run_with_mpi=pool_size > 1,
).run()

SimulateObs(
    progress=True,
    is_run_with_mpi=pool_size > 1,
).run()

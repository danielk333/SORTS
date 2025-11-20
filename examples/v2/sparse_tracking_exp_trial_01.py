import logging, time, typing as t, pickle, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import numpy.typing as npt
import xarray as xr
from astropy.time import Time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import sorts
from sorts import interpolation, population, propagator, radar, ExperimentDetail
from sorts.types import Tuple_3, Tuple_7
from sorts.space_object import SpaceObject, SpaceObjectId
from sorts.schedule.priority_scheduling import priority_scheduling
from sorts.controller import SparseTrackerController
from sorts.simulation.funcs import (
    ensure_directory_exist,
    safe_pickle,
    duplicate_and_perturbate_space_objects,
)
from sorts.simulation.stx_mrx_simulation import (
    stx_mrx_simulation,
    StxMrxSimulation,
    SimulationUnit,
    SimulationUnitState,
    Observation,
)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING) # suppress matplotlib logs below "warning"; fmt: skip;
logging.getLogger("sorts.propagator").setLevel(logging.WARNING)
logging.getLogger("sorts.frames").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info("starting example")

matplotlib.use("Agg")  # Use a non-GUI backend

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir", type=Path, default=Path(__file__).parent / ".." / ".." / "local_data"
)
parser.add_argument(
    "--master_catalog",
    type=Path,
    default=Path(__file__).parent / ".." / ".." / "local_data" / "celn_20090501_00.sim",
    dest="catalog_fpath",
)
args = parser.parse_args()

_SuK = SimulationUnit._K

spobj_dname_tpl = "spobj.{id}"
pert_dname_tpl = "pert.{id}"


# TODO: we need a sampler class;
#   this dsec_sampler func is move to top level because pickle won't work otherwise;
#   class should work better with pickle;
def dsec_sampler(orbit, epoch, start_time, end_time):
    dt = (end_time - start_time) / np.timedelta64(1, "s")
    t0 = (start_time - epoch) / np.timedelta64(1, "s")
    return np.arange(t0, t0 + dt, 120, dtype=np.float64)


def calc_jacobian(
    true_spobj: SpaceObject,
    pert_spobjs: list[SpaceObject],
    obs_jaco_tuple_multistatic_set: t.Sequence[Tuple_7[Observation]],
) -> npt.NDArray[np.float64]:
    multistatic_size = len(obs_jaco_tuple_multistatic_set)
    idp_vars = true_spobj.state._cart[:, 0]  # independent variables

    # num of measurements
    num_meas = len(obs_jaco_tuple_multistatic_set[0][0].get_state_slice()[_SuK.multi_index])

    # TODO: remove, prints for debugging ---
    obs_idx = 0
    print(
        (
            "len(obs_state_jaco_tuple_multistatic_set[0][obs_idx].get_state_slice()[_SuK.time])",
            len(obs_jaco_tuple_multistatic_set[0][obs_idx].get_state_slice()[_SuK.multi_index]),
            len(obs_jaco_tuple_multistatic_set[1][obs_idx].get_state_slice()[_SuK.multi_index]),
            len(obs_jaco_tuple_multistatic_set[2][obs_idx].get_state_slice()[_SuK.multi_index]),
        )
    )
    print("obs_state_jaco_tuple_multistatic_set[0][obs_idx].passage")
    print(obs_jaco_tuple_multistatic_set[0][obs_idx].passage)
    print(obs_jaco_tuple_multistatic_set[1][obs_idx].passage)
    print(obs_jaco_tuple_multistatic_set[2][obs_idx].passage)
    # ---

    if any(
        [
            len(obs_state_jaco_tuple[0].get_state_slice()[_SuK.multi_index]) != num_meas
            for obs_state_jaco_tuple in obs_jaco_tuple_multistatic_set
        ]
    ):
        raise RuntimeError(
            "multistatic measurement assumption violated;"
            + 'the true observation in the jacobian tuple (i.e. the 0th item) across the "multistatic_set" '
            + "should have the same size."
        )

    # init the jacobian
    J = np.zeros([num_meas * 2 * multistatic_size, len(idp_vars)], dtype=np.float64)

    for multistatic_idx, obs_jaco_tuple in enumerate(obs_jaco_tuple_multistatic_set):
        true_obs, *pert_obs = obs_jaco_tuple

        true_obs_state = true_obs.get_state_slice()
        pert_obs_states = [obs.get_state_slice() for obs in pert_obs]

        r_orig = true_obs_state[_SuK.two_way_range].to_numpy()
        v_orig = true_obs_state[_SuK.two_way_range_rate].to_numpy()

        for pert_idx, x_orig in enumerate(idp_vars):
            x_pert = pert_spobjs[pert_idx].state._cart[pert_idx, 0]
            r_pert = pert_obs_states[pert_idx][_SuK.two_way_range].to_numpy()
            v_pert = pert_obs_states[pert_idx][_SuK.two_way_range_rate].to_numpy()

            j_r_stt = multistatic_idx * num_meas * 2
            j_r_end = j_r_stt + num_meas
            j_v_stt = j_r_end
            j_v_end = j_v_stt + num_meas

            J[j_r_stt:j_r_end, pert_idx] = (r_pert - r_orig) / (x_pert - x_orig)
            J[j_v_stt:j_v_end, pert_idx] = (v_pert - v_orig) / (x_pert - x_orig)

    return J


def calc_covariance_matrix(jacobian: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    J = jacobian
    num_rows = J.shape[0]

    r_stds_tx = 10.0
    v_stds_tx = 5.0

    Sigma_m_diag_elms = np.array(
        [r_stds_tx**2] * num_rows + [v_stds_tx**2] * num_rows, dtype=np.float64
    )
    Sigma_m_inv = np.diag(1.0 / Sigma_m_diag_elms)

    try:
        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)
    except np.linalg.LinAlgError:
        Sigma_orb = np.full((6, 6), np.nan, dtype=np.float64)

    return t.cast(npt.NDArray[np.float64], Sigma_orb)


class WParam(t.TypedDict):
    param: SimulationUnit.FromPassagesOverTxRxStationPairParam
    persist_fpath: Path


class MpiExample(sorts.MpiQueuedExecution):
    sim_unit_fname_tpl = stx_mrx_simulation.sim_unit_fname_tpl

    def master_process(self):
        ##
        # prepare simulation environment
        ##

        save_dname = f"[{datetime.now().replace(microsecond=0).isoformat(sep=" ").replace(":", ".").replace("-", ".")}Z] sparse_tracking_exp_trial_01"
        save_dpath = args.out_dir / save_dname
        ensure_directory_exist(save_dpath)

        start_time = Time("2025-01-01 00:00:00")
        # end_time = Time("2025-01-01 03:00:00")
        end_time = Time("2025-01-02 00:00:00")  # remove: adj for debugging
        control_slice_duration = np.timedelta64(100_000, "us")  # 100ms
        coherent_integration_time = 0.04

        radar_sys = sorts.radar.radars.nostra.gen_nostra(
            frequency=3.2e9,
            antenna_num=10_000,
            antenna_spacing_lambda=0.5,
            antenna_efficiency=0.6,
            antenna_input_power=158,  # W
            thermal_load=66.0,  # W
            noise_figure_db=0.7,
            amplifier_gain_db=18,
            insertion_loss_db=0.35,
            aperture_efficiency=0.4,
            duty_cycle=0.2,
            t_sky=10.0,
            coherent_integration_time=coherent_integration_time,
            bandwidth_reduction_to_downsampling_ratio=10,
        )
        # TODO: these patching of station prop should be integrated into codebase
        tx_station: radar.Station = radar_sys.tx[0]
        tx_station.uid = 0
        rx_station_0: radar.Station = radar_sys.rx[0]
        rx_station_0.uid = 1
        rx_station_1: radar.Station = radar_sys.rx[1]
        rx_station_1.uid = 2
        rx_station_2: radar.Station = radar_sys.rx[2]
        rx_station_2.uid = 3

        _spobj_pop = population.master_catalog(
            args.catalog_fpath,
            mjd0=t.cast(float, start_time.mjd),
            propagator=propagator.SGP4,
            propagator_options={"settings": {"in_frame": "TEME", "out_frame": "ITRF"}},
        )
        rand_seed = 1203
        spobj_pop = population.master_catalog_factor(_spobj_pop, treshhold=1e-2, seed=rand_seed)
        rng = np.random.default_rng(seed=rand_seed)
        oids = rng.choice(len(spobj_pop), 5, replace=False)
        # spobjs = [spobj_pop.get_object(i) for i in oids]
        spobjs = [
            spobj_pop.get_object(oid) for oid in oids if oid == 33244
        ]  # remove: adj for debugging

        tracker_ctrls = [
            SparseTrackerController.from_space_object(
                SparseTrackerController.FromSpaceObjectParam(
                    tx_station=tx_station,
                    rx_stations=[rx_station_0, rx_station_1, rx_station_2],
                    exp_detail=ExperimentDetail(
                        id=exp_id,
                        # not used
                        coh_int_bandwidth=1.0,
                        ipp=1.0,
                        pulse_length=1.0,
                        duty_cycle=1.0,
                        # --
                        power=tx_station.power,
                        bandwidth=1 / coherent_integration_time,
                        noise_temp=rx_station_0.noise,
                        slice_duration=control_slice_duration,
                    ),
                    space_object=spobj,
                    epoch=start_time,
                    points_per_passage=5,
                )
            )
            for exp_id, spobj in enumerate(spobjs)
        ]

        pbar = tqdm(desc="Creating controllers", total=len(spobjs))
        tracker_schs = []
        for tracker_ctrl in tracker_ctrls:
            tracker_schs.append(tracker_ctrl.generate(start_time, end_time))
            pbar.update(1)
        pbar.close()
        del pbar  # otherwise will break pickle

        exp_id_stn_id_pairs_map = {}
        for tracker_ctrl in reversed(tracker_ctrls):
            exp_id_stn_id_pairs_map.update(tracker_ctrl.get_experiment_id_station_id_pairs_map())

        master_sch = priority_scheduling(tracker_schs, exp_id_stn_id_pairs_map)

        # TODO: probably better to make it an explicit dict instead of calling `locals()`
        # converted to dict to make it slightly safer
        sim_env = dict(locals())
        safe_pickle(sim_env, save_dpath / "sim_env.pickle")

        # a dict to hold the pickle fpath of 7 sim_units (true x1 + pert x6) of each spobj for easier result analysis later
        sim_unit_fpath_grp_by_pert: dict[SpaceObjectId, dict[int, list[Path]]] = {}

        # repeat the simulation for all duplicates from perturbation
        spobj_jacobian_tuples = duplicate_and_perturbate_space_objects(spobjs)
        spobj_grp_by_pert = list(zip(*spobj_jacobian_tuples)) # i.e. len 7, [true_spobj_list, pert_spobj_list...x6]; fmt: skip;
        for pert_idx, spobj_grp in enumerate(spobj_grp_by_pert):
            sim = StxMrxSimulation.from_controllers(
                controllers=tracker_ctrls,
                schedule=master_sch,
                epoch=start_time,
                start_time=start_time,
                end_time=end_time,
                space_objects=spobj_grp,
                dsec_sampler=dsec_sampler,
                interpolator_class=interpolation.Legendre8,
                # interpolator_class=interpolation.Linear,
            )
            safe_pickle(sim, save_dpath / "sim" / pert_dname_tpl.format(id=pert_idx) / "sim.pickle")

            sim_units_params = sim.prepare_simulation_unit_params()

            ##
            # invoke `mpi_master_proc_loop` to dispatch jobs to mpi worker process
            ##
            calc_start_time = time.perf_counter()

            job_params: list[WParam] = [
                {
                    "param": sim_units_param,
                    "persist_fpath": save_dpath
                    / spobj_dname_tpl.format(id=sim_units_param.spobj.oid)
                    / pert_dname_tpl.format(id=pert_idx)
                    / self.sim_unit_fname_tpl.format(id=sim_units_param.id),
                }
                for sim_units_param in sim_units_params
            ]

            for job_param in job_params:
                spobj_id = job_param["param"].spobj.oid
                if spobj_id not in sim_unit_fpath_grp_by_pert:
                    sim_unit_fpath_grp_by_pert[spobj_id] = {}

                if pert_idx not in sim_unit_fpath_grp_by_pert[spobj_id]:
                    sim_unit_fpath_grp_by_pert[spobj_id][pert_idx] = [job_param["persist_fpath"]]
                else:
                    sim_unit_fpath_grp_by_pert[spobj_id][pert_idx].append(job_param["persist_fpath"]) # fmt: skip

            self.mpi_master_proc_loop(job_params)
            # note to self: barrier here for finish sim

            calc_time = time.perf_counter() - calc_start_time
            logger.info(f"mpi_master_proc_loop took {calc_time} sec")

        ##
        # analyze result
        ##

        # the analysis will mostly use the group of space objects without perturbation,
        # the perturbated groups will be used for jacobian calculation
        calc_start_time = time.perf_counter()

        obss: list[stx_mrx_simulation.Observation] = []

        for spobj_id in sim_unit_fpath_grp_by_pert:

            # the set of sim_units for a multistatic radar, without perturbation
            true_sim_unit_multistatic_set: list[SimulationUnit] = []
            for fpath in sim_unit_fpath_grp_by_pert[spobj_id][0]:
                with open(fpath, "rb") as f:
                    true_sim_unit_multistatic_set.append(pickle.load(f))

            # a list of 6 sets of sim_units for a multistatic radar, with perturbation
            pert_sim_unit_multistatic_sets: list[list[SimulationUnit]] = []
            for i in range(6):
                pert_sim_unit_multistatic_sets.append([])
                for fpath in sim_unit_fpath_grp_by_pert[spobj_id][i + 1]:
                    with open(fpath, "rb") as f:
                        pert_sim_unit_multistatic_sets[i].append(pickle.load(f))

            spobj = true_sim_unit_multistatic_set[0].space_object
            pert_spobjs = [su[0].space_object for su in pert_sim_unit_multistatic_sets]

            # gather the observations into a list of tuples of 7 for jacobian calculation
            obs_jaco_tuple_per_multistatic_source: list[list[Tuple_7[Observation]]] = [
                [] for _ in range(len(true_sim_unit_multistatic_set))
            ]
            for multistatic_idx, sim_units in enumerate(
                zip(true_sim_unit_multistatic_set, *pert_sim_unit_multistatic_sets)
            ):
                # redeclared the interation var to add type info, and then unpack it
                sim_units: Tuple_7[SimulationUnit] = sim_units
                sim_unit, *pert_sim_units = sim_units

                obss.extend(sim_unit.observations)

                obs_jaco_tuple_per_multistatic_source[multistatic_idx] = list(
                    zip(sim_unit.observations, *(su.observations for su in pert_sim_units))
                )

            # TODO: `obs_jaco_tuple_multistatic_set_list` need better naming, and recheck logic for safety?
            obs_jaco_tuple_multistatic_set_list: list[Tuple_3[Tuple_7[Observation]]] = list(
                zip(*obs_jaco_tuple_per_multistatic_source)
            )

            for obs_jaco_tuple_multistatic_set in obs_jaco_tuple_multistatic_set_list:
                # TODO: add back snr threshold filter
                # TODO: min `_SuK.multi_index` len filtering? (e.g. 1 or 5)
                # calc the jacobian
                J = calc_jacobian(
                    true_spobj=spobj,
                    pert_spobjs=pert_spobjs,
                    obs_jaco_tuple_multistatic_set=obs_jaco_tuple_multistatic_set,
                )
                logger.info(f"jacobian: {J}")  # TODO: just a place holder usage of the jacobian; fmt: skip;

                # calc covariance matrix for error estimation of linearized orbit determination
                Sigma_orb = calc_covariance_matrix
                logger.info(f"Sigma_orb: {Sigma_orb}")  # TODO: just a place holder usage of the Sigma_orb; fmt: skip;

            print(f"len(obss): {len(obss)}")

            calc_time = time.perf_counter() - calc_start_time
            logger.info(f"result_analysis_fn took {calc_time} sec")

        return

    def worker_process(self, job_param):
        job_param = t.cast(WParam, job_param)

        param = job_param["param"]
        persist_fpath = job_param["persist_fpath"]
        worker_proc_rank = self.rank

        try:
            if persist_fpath.exists():
                logger.info(
                    f"worker: {worker_proc_rank} | SimulationUnit: {param.id} already completed, will load from the saved file instead"
                )

                with open(persist_fpath, "rb") as f:
                    sim_unit = pickle.load(f)

            else:
                # NOTE: sim_unit is saved 2 times, 1 before running `simulate` and 1 after

                sim_unit = SimulationUnit.from_passages_over_tx_rx_station_pair(param)

                ensure_directory_exist(persist_fpath.parent)
                safe_pickle(sim_unit, persist_fpath)
                logger.info(f"worker: {worker_proc_rank} | `SimulationUnit.simulate` start")
                sim_unit.simulate()

                # delete the file we saved earlier before saving again
                persist_fpath.unlink(missing_ok=True)
                safe_pickle(sim_unit, persist_fpath)

        except Exception as err:
            raise RuntimeError(
                f"Runtime fail in worker: {worker_proc_rank} | SimulationUnit: {param.id}"
            ) from err

        obss = sim_unit.observations

        self.comm.send(True, dest=self.master_proc_rank)
        logger.info(
            f"worker: {worker_proc_rank} | SimulationUnit:{sim_unit.id} done with {len(obss)} observations"
        )


execution = MpiExample(
    is_run_with_mpi=False,  # a convenience flag to switch between running mode for debugging
    # is_run_with_mpi = True  # a convenience flag to switch between running mode for debugging
).run()

exit()

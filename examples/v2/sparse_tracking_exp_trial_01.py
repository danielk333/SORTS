import logging, time, typing as t, pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
from astropy.time import Time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sorts
from sorts import types, interpolation, population, propagator, radar
from sorts.space_object import SpaceObject
from sorts.schedule.priority_scheduling import priority_scheduling
from sorts.controller import SparseTrackerController
from sorts.simulation.funcs import ensure_directory_exist, safe_pickle
from sorts.simulation.stx_mrx_simulation import stx_mrx_simulation, StxMrxSimulation, SimulationUnit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("starting example")

matplotlib.use("Agg")  # Use a non-GUI backend


# TODO: we need a sampler class;
#   this dsec_sampler func is move to top level because pickle won't work otherwise;
#   class should work better with pickle;
def dsec_sampler(orbit, start_time, end_time):
    return np.arange(0, (end_time - start_time) / np.timedelta64(1, "s"), 120, dtype=np.float64)


class WParam(t.TypedDict):
    param: SimulationUnit.FromPassagesOverTxRxStationPairParam
    persist_dpath: Path


class MpiExample(sorts.MpiQueuedExecution):
    sim_unit_fname_tpl = stx_mrx_simulation.sim_unit_fname_tpl

    def master_process(self):
        ##
        # prepare simulation environment
        ##

        save_dname = f"[{datetime.now().replace(microsecond=0).isoformat(sep=" ").replace(":", ".").replace("-", ".")}Z] mpi"
        save_dpath = Path(__file__).parent / ".." / ".." / "local_data" / save_dname
        ensure_directory_exist(save_dpath)

        start_time = Time("2025-01-01 02:45:00")
        end_time = Time("2025-01-01 03:00:00")
        control_slice_duration = np.timedelta64(10_000, "us")  # 10ms

        radar_sys = sorts.get_radar("nostra", "example1")
        # TODO: these patching of station prop should be integrated into codebase
        tx_station: radar.Station = radar_sys.tx[0]
        tx_station.uid = 0
        rx_station_0: radar.Station = radar_sys.rx[0]
        rx_station_0.uid = 1
        rx_station_1: radar.Station = radar_sys.rx[1]
        rx_station_1.uid = 2

        known_spobj = SpaceObject(
            oid=-1,
            propagator=propagator.SGP4,
            propagator_options={"settings": {"out_frame": "ITRF"}},
            a=7200e3,
            e=0.02,
            i=75,
            raan=86,
            aop=0,
            mu0=60,
            epoch=start_time,
            parameters={"d": 0.1},
        )

        catalog_fpath = Path(__file__).parent / ".." / ".." / "local_data" / "celn_20090501_00.sim"
        _spobj_pop = population.master_catalog(
            catalog_fpath,
            propagator=propagator.SGP4,
            propagator_options={"settings": {"in_frame": "TEME", "out_frame": "ITRF"}},
        )
        rand_seed = 120389
        spobj_pop = population.master_catalog_factor(_spobj_pop, treshhold=1.0, seed=rand_seed)
        spobjs = [
            known_spobj,
            *[spobj_pop.get_object(i) for i in range(spobj_pop.shape[0])][0:10],
        ]

        tracker_ctrls = [
            SparseTrackerController.from_space_object(
                SparseTrackerController.FromSpaceObjectParam(
                    tx_station=tx_station,
                    rx_stations=[rx_station_0, rx_station_1],
                    exp_detail={
                        "id": exp_id,
                        "coh_int_bandwidth": 1.0,
                        "ipp": 1.0,
                        "pulse_length": 1.0,
                        "power": 5000000.0,
                        "bandwidth": 52.08333333333333,
                        "duty_cycle": 1.0,
                        "noise_temp": 150.0,
                        "slice_duration": control_slice_duration,
                    },
                    space_object=spobj,
                    epoch=start_time,
                    points_per_passage=5,
                )
            )
            for exp_id, spobj in enumerate(spobjs)
        ]

        tracker_schs = [
            tracker_ctrl.generate(start_time, end_time) for tracker_ctrl in tracker_ctrls
        ]

        exp_id_stn_id_pairs_map = {}
        for tracker_ctrl in reversed(tracker_ctrls):
            exp_id_stn_id_pairs_map.update(tracker_ctrl.get_experiment_id_station_id_pairs_map())

        master_sch = priority_scheduling(tracker_schs, exp_id_stn_id_pairs_map)

        spec_by_controllers: stx_mrx_simulation.SpecByControllers = {
            "controllers": tracker_ctrls,
            "schedule": master_sch,
            "epoch": start_time,
            "start_time": start_time,
            "end_time": end_time,
            "space_objects": spobjs,
            "dsec_sampler": dsec_sampler,
            # "interpolator_class": interpolation.Legendre8,
            "interpolator_class": interpolation.Linear,
        }

        # TODO: probably better to make it an explicit dict instead of calling `locals()`
        # converted to dict to make it slightly safer
        sim_env = dict(locals())
        safe_pickle(sim_env, save_dpath / "sim_env")

        sim = StxMrxSimulation.from_controllers(spec_by_controllers)
        sim_units_params = sim.prepare_simulation_unit_params()

        ##
        # invoke `mpi_master_proc_loop` to dispatch jobs to mpi worker process
        ##
        calc_start_time = time.perf_counter()

        job_params: list[WParam] = [
            {
                "param": sim_units_param,
                "persist_dpath": save_dpath,
            }
            for sim_units_param in sim_units_params
        ]
        self.mpi_master_proc_loop(job_params)

        calc_time = time.perf_counter() - calc_start_time
        logger.info(f"mpi_master_proc_loop took {calc_time} sec")

        ##
        # analyze result
        ##
        calc_start_time = time.perf_counter()

        obss: list[stx_mrx_simulation.Observation] = []
        max_snrs_value = []
        max_snrs_time: list[types.Datetime64_us] = []
        max_snrs_spobj_id: list[int] = []

        for sim_unit in stx_mrx_simulation.iter_mpi_simulation_results(save_dpath):
            logger.info(f"processing result from SimulationUnit <{sim_unit.id}>")

            _SuK = SimulationUnit._K

            su_obss = sim_unit.observations
            obss.extend(su_obss)

            for obs in su_obss:
                obs_state = obs.get_state_slice()
                argmax_snr = t.cast(xr.DataArray, obs_state[_SuK.snr].argmax())
                midx_max_snr = obs_state[{_SuK.multi_index: argmax_snr.item()}]
                midx_max_snr_value = midx_max_snr[_SuK.snr].item()
                midx_max_snr_time = midx_max_snr[_SuK.time].item()

                max_snrs_value.append(midx_max_snr_value)
                max_snrs_time.append(midx_max_snr_time)
                max_snrs_spobj_id.append(sim_unit.space_object.oid)

        # plotting
        logger.info(f"start generating plots...")

        fig, ax = plt.subplots()
        ax.set_title("snr vs time")
        ax.scatter(max_snrs_time, max_snrs_value, s=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        fig.autofmt_xdate()
        # ax.set_yscale("log")
        plt.savefig(save_dpath / "max_snr_vs_time.png", dpi=300, bbox_inches="tight")

        logger.info(f"done generating plots")

        print(f"len(obss): {len(obss)}")

        calc_time = time.perf_counter() - calc_start_time
        logger.info(f"result_analysis_fn took {calc_time} sec")

        return

    def worker_process(self, job_param):
        param = job_param["param"]
        persist_dpath = job_param["persist_dpath"]
        persist_fname = self.sim_unit_fname_tpl.format(id=param.id)
        persist_fpath = persist_dpath / persist_fname
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

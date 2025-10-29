import logging, typing as t, functools, operator
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
from astropy.time import Time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sorts
from sorts import (
    types,
    interpolation,
    population,
    propagator,
    radar,
)
from sorts.space_object import SpaceObject
from sorts.schedule.priority_scheduling import priority_scheduling
from sorts.controller import SparseTrackerController
from sorts.simulation.stx_mrx_simulation import (
    stx_mrx_simulation,
    StxMrxSimulation,
    SimulationUnit,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("starting example")

matplotlib.use("Agg")  # Use a non-GUI backend


# TODO: we need a sampler class;
#   this dsec_sampler func is move to top level because pickle won't work otherwise;
#   class should work better with pickle;
def dsec_sampler(orbit, start_time, end_time):
    return np.arange(0, (end_time - start_time) / np.timedelta64(1, "s"), 120, dtype=np.float64)


def duplicate_and_perturbate_space_objects(spobjs: list[SpaceObject], dup_num=6):
    # duplicate list items by nesting and then flattening;
    spobjs = list(functools.reduce(operator.concat, [[spobj] * dup_num for spobj in spobjs]))

    # TODO: perturbation

    return spobjs


class MpiExample(sorts.MpiQueuedExecution):
    def prepare_master_process_environment(self):
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

        # prep for jacobian calculation
        spobjs = duplicate_and_perturbate_space_objects(spobjs)

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

        return sim_env

    def create_simulation(self, menv):
        return StxMrxSimulation.from_controllers(menv["spec_by_controllers"])

    def run_worker_job(self, persist_dpath, jab_param):
        param = jab_param["param"]
        persist_dpath = jab_param["persist_dpath"]

        stx_mrx_simulation.mpi_worker_job(
            comm=self.comm,
            master_proc_rank=self.master_proc_rank,
            worker_proc_rank=self.rank,
            persist_dpath=persist_dpath,
            param=param,
        )

    def analyze_result(self, menv, sim, persist_dpath):
        max_snrs_value = []
        max_snrs_time: list[types.Datetime64_us] = []
        max_snrs_spobj_id: list[int] = []

        for sim_unit in stx_mrx_simulation.iter_mpi_simulation_results(persist_dpath):
            logger.info(f"processing result from SimulationUnit <{sim_unit.id}>")

            _SuK = SimulationUnit._K

            obss = sim_unit.observations
            for obs in obss:
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
        ax.set_yscale("log")
        plt.savefig(persist_dpath / "max_snr_vs_time.png", dpi=300, bbox_inches="tight")

        logger.info(f"done generating plots")


dname = f"[{datetime.now().replace(microsecond=0).isoformat(sep=" ").replace(":", ".").replace("-", ".")}Z] mpi"
execution = MpiExample(
    sim_unit_fname_tpl=stx_mrx_simulation.sim_unit_fname_tpl,
).run(
    persist_dpath=Path(__file__).parent / ".." / ".." / "local_data" / dname,
    # is_run_with_mpi=True,  # a convenience flag to switch between running mode for debugging
    is_run_with_mpi=False,  # a convenience flag to switch between running mode for debugging
)

exit()

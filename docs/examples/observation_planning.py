import logging, typing as t
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.time import Time
from mpi4py import MPI
from tqdm import tqdm
import spacecoords
import sorts
from sorts import utils, pointing, population, radar, passage, schedule, simulation

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("sorts.propagator").setLevel(logging.WARNING)
logging.getLogger("sorts.frames").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info("starting example")

R_earth = 6371e3


class Script(sorts.MpiJobQueueExecutor):
    # NOTE: code here (outside of methods) are ran in all processes of MPI

    # persistence configs
    data_dpath = Path(__file__).parent / "observation_planning_data"
    clobber = True

    tle_fpath = data_dpath / "example.tle"

    # config random seed
    rand_seed = 1203
    np.random.seed(rand_seed)
    rng = np.random.default_rng(seed=rand_seed)

    start_time = Time("2025-01-01 00:00:00")
    end_time = Time("2025-01-02 00:00:00")

    time_step = 10.0
    coherent_integration_time = 0.04
    duty_cycle = 0.2
    time_slice = coherent_integration_time / duty_cycle
    control_slice_duration = np.timedelta64(int(time_slice * 1e6), "us")

    radar_sys = radar.radars.nostra.gen_nostra(
        frequency=3.2e9,
        antenna_num=10_000,
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
        station_uid_map={"se_tx": 0, "no_tx": 1, "fi_tx": 2, "se_rx": 3, "no_rx": 4, "fi_rx": 5},
    )

    exp_id = 0
    radar_power = radar_sys.tx[0].power
    radar_bandwidth = 1 / coherent_integration_time
    radar_noise_temp = radar_sys.rx[0].noise

    radar_detection_system = simulation.RadarDetectionSystem.from_stations(
        tx_station=radar_sys.tx[0],
        rx_stations=radar_sys.rx,
    )

    spobj_motion_sys = sorts.SpaceObjectMotionSystem(
        propagator=sorts.propagator.Sgp4(
            settings=sorts.propagator.Sgp4Settings(out_frame="ITRS", mean_elements_input=True)
        ),
        interpolator=sorts.interpolation.Legendre8,
    )

    def master_main(self):
        utils.ensure_directory_exist(self.data_dpath)

        spobj_pop = population.tle_catalog(
            self.tle_fpath,
            save_mean_elements=True,
            degrees=True,
        )

        oids = np.arange(len(spobj_pop))
        spobjs = [spobj_pop.get_object(oid) for oid in oids]

        # propagate
        params_list = [
            (
                self.data_dpath
                / f"space_object_{spobj.object_id}"
                / "propagate_step_output.pickle",
                self.clobber,
                spobj,
            )
            for spobj in spobjs
        ]

        propagate_step_pickle_list = self.run_job_queue(
            worker_process=self.propagate,
            job_params_list=params_list,
        )

        # compute_schedule_and_passages step
        params_list = [
            (
                self.data_dpath
                / f"space_object_{spobj.object_id}"
                / "schedule_and_passages.pickle",
                self.clobber,
                propagate_step_pickle,
            )
            for spobj, propagate_step_pickle in zip(spobjs, propagate_step_pickle_list)
        ]
        schedule_and_passages_pickle_list = self.run_job_queue(
            worker_process=self.compute_schedule_and_passages,
            job_params_list=params_list,
        )

        # per object analysis step
        logger.debug("starting per object analysis")

        params_list = [
            (
                self.data_dpath / f"space_object_{spobj.object_id}" / "analysis_result.pickle",
                self.clobber,
                propagate_step_pickle,
                schedule_and_passages_pickle,
            )
            for spobj, propagate_step_pickle, schedule_and_passages_pickle in zip(
                spobjs, propagate_step_pickle_list, schedule_and_passages_pickle_list
            )
        ]
        analysis_results = self.run_job_queue(
            worker_process=self.analyse_passages, job_params_list=params_list
        )

        # summary of per object analysis step
        logger.debug("starting summary of per object analysis")

        sch_crossing_slice_list: list[pd.DataFrame] = []
        for analysis_result in analysis_results:
            spobj, sch, crossings_mask = utils.as_retval(
                self.analyse_passages, analysis_result
            ).load()
            sch_crossing_slice_list.append(sch.loc[crossings_mask])

        sch_crossings = pd.concat(sch_crossing_slice_list)
        print(sch_crossings)

        logger.debug("planning done")

    @utils.use_pickled_or_compute_method
    def propagate(self, spobj: sorts.SpaceObject):
        times, spobj_states = self.spobj_motion_sys.propagate(
            space_object=spobj,
            start_time=self.start_time,
            end_time=self.end_time,
            time_step=self.time_step,
        )

        spobj_interp = self.spobj_motion_sys.make_interpolation(times=times, states=spobj_states)

        return (times, spobj, spobj_states, spobj_interp)

    @utils.use_pickled_or_compute_method
    def compute_schedule_and_passages(self, propagate_step_pickle: utils.PickledObject):
        (
            times,
            spobj,
            spobj_states,
            spobj_interp,
        ) = utils.as_retval(self.propagate, propagate_step_pickle).load()

        passages = passage.find_simultaneous_passages(
            dt=times,
            space_object=spobj,
            states=spobj_states[:3, ...],
            tx_station=self.radar_detection_system.tx_station,
            rx_stations=self.radar_detection_system.rx_stations,
            epoch=utils.to_datetime64_us(self.start_time),
        )

        tracker_sch = pointing.sparse_tracking(
            passages_of_spobj=passages,
            interpolation=spobj_interp,
            points_per_passage=10,
            tx_station=self.radar_detection_system.tx_station,
            rx_stations=self.radar_detection_system.rx_stations,
            exp_id=self.exp_id,
            slice_duration=self.control_slice_duration,
        )

        return tracker_sch, passages

    @utils.use_pickled_or_compute_method
    def analyse_passages(
        self,
        propagate_step_pickle: utils.PickledObject,
        schedule_and_passages_pickle: utils.PickledObject,
    ):
        (
            times,
            spobj,
            spobj_states,
            spobj_interp,
        ) = utils.as_retval(self.propagate, propagate_step_pickle).load()

        sch, passages = utils.as_retval(
            self.compute_schedule_and_passages, schedule_and_passages_pickle
        ).load()

        _K = schedule.ScheduleKey
        enus = sch[[_K.pointing_e, _K.pointing_n, _K.pointing_u]].to_numpy().T
        # NOTE: `r` in `azelrs` are mostly/all `1`s because it is converted from pointings that has no "range" info
        azelrs = spacecoords.spherical.cart_to_sph(enus, degrees=True)

        # check when `az` component crosses zero; the start of crossing is marked.
        azs = pd.Series(azelrs[0])
        crossings_mask = (azs.shift(-1) * azs) < 0

        # insert a spobj id column into sch
        sch["spobj_id"] = spobj.object_id
        sch_with_spobj_id = sch

        return spobj, sch_with_spobj_id, crossings_mask


Script(is_run_with_mpi=MPI.COMM_WORLD.Get_size() > 1).run()

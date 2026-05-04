from __future__ import annotations
import logging, typing as t, traceback, time, enum, dataclasses, sys, functools
from tqdm import tqdm
from mpi4py import MPI

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MpiJobQueueExecutor:
    """
    Use MPI as a job queue, with the 0th rank process as master and the rest as workers.
    """

    master_proc_rank: t.ClassVar[t.Final] = 0

    num_workers: int
    comm: MPI.Intracomm = dataclasses.field(default_factory=lambda: MPI.COMM_WORLD)
    is_run_with_mpi: bool = True

    def run_job_queue[JobParams](
        self,
        job_params_list: t.Sequence[JobParams],
        worker_process: t.Callable[[JobParams], None],
        show_progress_bar=True,
    ):

        if self.is_run_with_mpi:
            # Run with MPI, the normal path.
            try:
                # master
                if self.comm.rank == self.master_proc_rank:
                    self.mpi_master_loop(job_params_list, show_progress_bar)

                # workers
                else:
                    self.mpi_worker_loop(worker_process)

            except Exception as err:
                comm = MPI.COMM_WORLD
                r = comm.Get_rank()

                logger.error(
                    f"terminating MPI process due to exception occured in rank: {r}, error:\n"
                    + "\n".join(traceback.format_exception(err))
                )
                comm.Abort(1)

        else:
            # Run the execution without MPI, mostly useful for debugging.
            calc_start_time = time.perf_counter()

            pbar = None
            if show_progress_bar:
                pbar = tqdm("Worker progress", total=len(job_params_list), file=sys.stdout)

            for work_job_param in job_params_list:
                worker_process(work_job_param)

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            calc_time = time.perf_counter() - calc_start_time
            logger.info(f"run_job_queue done, took {calc_time} sec")

    # TODO: add a way to spawn mpi process from python
    #     subprocess.run(["mpiexec", "-n", str(num_workers), "python", sys.argv[0]])

    def mpi_master_loop[JobParams](
        self, job_params_list: t.Sequence[JobParams], show_progress_bar=True
    ) -> None:
        calc_start_time = time.perf_counter()

        _K = MpiQueuedExecutorKey

        ##
        # parallization section
        ##
        logger.info(f"master: {self.master_proc_rank} | parallel processing of worker jobs start")

        is_worker_idle_list = [True for _ in range(self.num_workers)]
        next_work_job_param_idx = 0
        processed_work_job_cnt = 0

        pbar = None
        if show_progress_bar:
            pbar = tqdm("MPI worker progress", total=len(job_params_list), file=sys.stdout)

        while processed_work_job_cnt < len(job_params_list):
            # send next work_job_param if there is idle worker
            if any(is_worker_idle_list) and next_work_job_param_idx < len(job_params_list):
                idle_worker_idx = is_worker_idle_list.index(True)
                idle_worker_rank = idle_worker_idx + 1

                job_params = job_params_list[next_work_job_param_idx]
                self.mpi_send(dest=idle_worker_rank, msg_type=_K.job_params, payload=job_params)

                logger.debug(
                    f"master: {self.master_proc_rank} | sent worker job"
                    + f" ({next_work_job_param_idx+1}/{len(job_params_list)}) to worker {idle_worker_rank}"
                )

                is_worker_idle_list[idle_worker_idx] = False
                next_work_job_param_idx += 1

            # otherwise, wait for result
            else:
                logger.debug(f"master: {self.master_proc_rank} | awaiting results ...")

                status = MPI.Status()
                self.mpi_recv(status=status)
                worker_rank = status.Get_source()

                processed_work_job_cnt += 1
                if show_progress_bar and pbar is not None:
                    pbar.update(1)

                is_worker_idle_list[worker_rank - 1] = True

        if show_progress_bar and pbar is not None:
            pbar.close()

        logger.info(f"master: {self.master_proc_rank} | master proc loop done, returning...")

        # TODO: maybe we can use `comm.bcast` here?
        #   but worker also need to call `comm.bcast` for listening,
        #   not sure if mpi allows listening to both `bcast` and `recv`
        for r in range(1, self.comm.size):  # start from 1 because 0 rank is the master
            logger.debug(f"master: {self.master_proc_rank} | terminating worker: {r} ...")
            self.mpi_send(dest=r, msg_type=_K.terminate)
            self.mpi_recv(source=r)  # wait for an ack

        calc_time = time.perf_counter() - calc_start_time
        logger.info(f"master_process took {calc_time} sec")

    def mpi_worker_loop[JobParams](self, worker_process: t.Callable[[JobParams], None]) -> None:
        """Run `worker_process` per `JobParams` from MPI comm until they are exhausted."""

        _K = MpiQueuedExecutorKey
        worker_proc_rank = self.comm.rank

        while True:
            logger.info(f"worker: {worker_proc_rank} | waiting for msg...")

            match self.mpi_recv(source=self.master_proc_rank):
                case MpiMsg(_K.terminate, _):
                    # exit if `_K.terminate` is received
                    logger.info(f"worker: {worker_proc_rank} | exiting...")
                    self.mpi_send(
                        dest=self.master_proc_rank, msg_type=_K.exit_ok
                    )  # reply an ack to master

                    # TODO: probably should not send `_K.terminate` if the worker will be reused
                    # break here to allow for further execution of workers later
                    break

                case MpiMsg(_K.job_params, payload):
                    job_param = t.cast(JobParams, payload)
                    try:
                        worker_process(job_param)
                        self.mpi_send(
                            dest=self.master_proc_rank, msg_type=_K.worker_process_return_ok
                        )
                    except BaseException as exc:
                        raise MpiQueuedExecutorError(
                            f"worker: {worker_proc_rank} | Error during job:\n {job_param}"
                        ) from exc

                case msg:
                    # throw for unexpected msg
                    raise RuntimeError(
                        f"worker: {worker_proc_rank} | received unexcepted msg: {msg}"
                    )

    def master_only[**Params, Ret](self, func: t.Callable[Params, Ret]):
        """
        A function decorator that makes it run in the master process (MPI rank 0) only.

        The return type of the noop code path is hidden (it returns `None`).

        NOTE:
            There is no `worker_only` counterpart,
            use the job dispatching mechanism of this class to pass job to workers instead.
        """

        @functools.wraps(func)
        def wrapper(*args: Params.args, **kwargs: Params.kwargs):
            # just a pass-through if we are not using MPI
            if not self.is_run_with_mpi:
                return func(*args, **kwargs)

            else:
                # Also a pass-through if we are a master rank MPI process
                if self.comm.rank == self.master_proc_rank:
                    return func(*args, **kwargs)
                else:
                    # return a noop func, and cast it to keep the original typings
                    noop_fn = t.cast(t.Callable[Params, Ret], lambda *args, **kwargs: None)
                    return noop_fn(*args, **kwargs)

        return wrapper

    def mpi_send(
        self,
        dest: int,
        msg_type: MpiQueuedExecutorKey,
        payload: object = None,
        tag: int = 0,
    ):
        """Send a `MpiMsg`."""

        self.comm.send(MpiMsg(msg_type, payload), dest, tag)

    def mpi_recv(self, *args, **kwargs):
        """Receive a `MpiMsg`."""

        return t.cast(MpiMsg, self.comm.recv(*args, **kwargs))


# TODO: remove?
@dataclasses.dataclass(kw_only=True, frozen=True)
class MpiProessInfo:
    rank: int


class MpiQueuedExecutorKey(enum.StrEnum):
    job_params = "job_params"
    exit_ok = "exit_ok"
    terminate = "terminate"
    worker_process_return_ok = "worker_process_return_ok"


class MpiQueuedExecutorError(Exception):
    pass


class MpiMsg[T](t.NamedTuple):
    type: MpiQueuedExecutorKey
    payload: T

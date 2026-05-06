from __future__ import annotations
import logging, typing as t, traceback, time, enum, dataclasses, sys, functools
from tqdm import tqdm
from mpi4py import MPI, typing as mpit

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

    def run_job_queue[*Args, Ret](
        self,
        job_params_list: t.Sequence[tuple[*Args]],
        worker_process: t.Callable[[*Args], Ret],
        show_progress_bar=True,
    ) -> list[Ret]:
        # Run with MPI, the normal path.
        if self.is_run_with_mpi:
            # master
            if self.comm.rank == self.master_proc_rank:
                return self.mpi_job_dispatching_loop(
                    job_params_list, worker_process, show_progress_bar
                )

            # workers
            else:
                raise RuntimeError(
                    f"MPI process rank {self.comm.rank} attempted to invoke `run_job_queue`, "
                    "but only MPI master process (i.e. rank 0) is allowed to do it."
                )

        # Run the execution without MPI (i.e. by looping), mostly useful for debugging.
        else:
            calc_start_time = time.perf_counter()

            pbar = None
            if show_progress_bar:
                pbar = tqdm("Worker progress", total=len(job_params_list), file=sys.stdout)

            worker_process_ret_list: list[Ret] = []
            for work_job_param in job_params_list:
                worker_process_ret_list.append(worker_process(*work_job_param))

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            calc_time = time.perf_counter() - calc_start_time
            logger.info(f"run_job_queue done, took {calc_time} sec")
            return worker_process_ret_list

    # TODO: add a way to spawn mpi process from python
    #     subprocess.run(["mpiexec", "-n", str(num_workers), "python", sys.argv[0]])

    def mpi_job_dispatching_loop[*Args, Ret](
        self,
        job_params_list: t.Sequence[tuple[*Args]],
        worker_process: t.Callable[[*Args], Ret],
        show_progress_bar=True,
    ) -> list:
        calc_start_time = time.perf_counter()

        ##
        # parallization section
        ##
        logger.info(f"master: {self.master_proc_rank} | parallel processing of worker jobs start")

        worker_ret_dict: dict[int, Ret] = {}
        is_worker_idle_list = [True for _ in range(self.num_workers)]
        next_job_params_idx = 0

        pbar = None
        if show_progress_bar:
            pbar = tqdm("MPI worker progress", total=len(job_params_list), file=sys.stdout)

        while next_job_params_idx < len(job_params_list):
            # send next work_job_param if there is idle worker
            if any(is_worker_idle_list) and next_job_params_idx < len(job_params_list):
                idle_worker_idx = is_worker_idle_list.index(True)
                idle_worker_rank = idle_worker_idx + 1

                args = job_params_list[next_job_params_idx]
                self.mpi_send(
                    msg=MpiMsgJobParams(next_job_params_idx, worker_process, args),
                    dest=idle_worker_rank,
                )

                logger.debug(
                    f"master: {self.master_proc_rank} | sent worker job"
                    + f" ({next_job_params_idx+1}/{len(job_params_list)}) to worker {idle_worker_rank}"
                )

                is_worker_idle_list[idle_worker_idx] = False
                next_job_params_idx += 1

            # otherwise, wait for result
            else:
                logger.debug(f"master: {self.master_proc_rank} | awaiting results ...")

                status = MPI.Status()
                match self.mpi_recv(status=status):
                    case MpiMsgWorkerProcessReturns(job_idx, _worker_process, _args, worker_ret):
                        worker_ret_dict[job_idx] = worker_ret
                        worker_rank = status.Get_source()

                        if show_progress_bar and pbar is not None:
                            pbar.update(1)

                        is_worker_idle_list[worker_rank - 1] = True

                    case msg:
                        # throw for unexpected msg
                        raise RuntimeError(
                            f"master: {self.master_proc_rank} | received unexcepted msg: {msg}"
                        )

        if show_progress_bar and pbar is not None:
            pbar.close()

        ret = [worker_ret_dict[k] for k in sorted(worker_ret_dict)]
        logger.info(f"master: {self.master_proc_rank} | master proc loop done, returning...")

        calc_time = time.perf_counter() - calc_start_time
        logger.info(f"master_process took {calc_time} sec")
        return ret

    def mpi_worker_loop(self):
        """
        Run `worker_process` for every `MpiMsg` message received.

        Stops when a `MpiMsgTerminate` message is received.
        """

        worker_proc_rank = self.comm.rank

        while True:
            logger.info(f"worker: {worker_proc_rank} | waiting for msg...")

            match self.mpi_recv(source=self.master_proc_rank):
                case MpiMsgTerminate():
                    logger.info(f"worker: {worker_proc_rank} | exiting...")
                    # reply an ack to master
                    self.mpi_send(msg=MpiMsgExitOk(), dest=self.master_proc_rank)

                    # TODO: probably should not send `MpiMsgTerminate` if the worker will be reused
                    # break here to allow for further execution of workers later
                    break

                case MpiMsgJobParams(job_idx, worker_process, args):
                    try:
                        ret = worker_process(*args)
                        self.mpi_send(
                            msg=MpiMsgWorkerProcessReturns(job_idx, worker_process, args, ret),
                            dest=self.master_proc_rank,
                        )
                    except BaseException as exc:
                        raise MpiQueuedExecutorError(
                            f"worker: {worker_proc_rank} | Error during job:\n {args}"
                        ) from exc

                case msg:
                    # throw for unexpected msg
                    raise RuntimeError(
                        f"worker: {worker_proc_rank} | received unexcepted msg: {msg}"
                    )

    # TODO: make `MpiJobQueueExecutor` an ABC and this as a virtual method?
    def entry_point[**Params, Ret](self, func: t.Callable[Params, Ret]):
        """
        A function decorator that make `func` the entry point of `MpiJobQueueExecutor`.

        The wrapped `func` will it run in the master process (MPI rank 0) only.

        In a worker process, a worker loop function from `MpiJobQueueExecutor` will be ran instead.
        The function signature of worker loop is hidden from the signature of this function decorator.
        """

        @functools.wraps(func)
        def wrapper(*args: Params.args, **kwargs: Params.kwargs):
            # just a pass-through if we are not using MPI
            if not self.is_run_with_mpi:
                return func(*args, **kwargs)

            else:
                try:
                    # Also a pass-through if we are a master rank MPI process
                    if self.comm.rank == self.master_proc_rank:
                        return func(*args, **kwargs)
                    else:
                        # return the worker loop otherwise
                        return t.cast(t.Callable[Params, Ret], self.mpi_worker_loop())

                except Exception as err:
                    r = self.comm.Get_rank()

                    logger.error(
                        f"terminating MPI process due to exception occured in rank: {r}, error:\n"
                        + "\n".join(traceback.format_exception(err))
                    )
                    self.comm.Abort(1)

        return wrapper

    # TODO: WIP
    def terminate(self):
        # TODO: maybe we can use `comm.bcast` here?
        #   but worker also need to call `comm.bcast` for listening,
        #   not sure if mpi allows listening to both `bcast` and `recv`
        for r in range(1, self.comm.size):  # start from 1 because 0 rank is the master
            logger.debug(f"master: {self.master_proc_rank} | terminating worker: {r} ...")
            self.mpi_send(msg=MpiMsgTerminate(), dest=r)
            self.mpi_recv(source=r)  # wait for an ack

    def master_only[**Params, Ret](self, func: t.Callable[Params, Ret]):
        """
        A function decorator that makes it run in the master process (MPI rank 0) only.

        For worker processes, `func` will be replaced by a noop func that returns `None`.
        The return type of this noop code path is hidden from the signature.

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
        msg: MpiMsg,
        dest: int,
        tag: int = 0,
    ):
        """Send a `MpiMsg`. `dest`, `tag` params are the same as `MPI.Intracomm.send`."""

        self.comm.send(msg, dest, tag)

    def mpi_recv[M: MpiMsg](
        self,
        buf: mpit.Buffer | None = None,
        source: int = MPI.ANY_SOURCE,
        tag: int = MPI.ANY_TAG,
        status: MPI.Status | None = None,
    ):
        """Receive a `MpiMsg`. Take the same params as `MPI.Intracomm.recv`."""

        return t.cast(M, self.comm.recv(buf=buf, source=source, tag=tag, status=status))


# TODO: remove?
@dataclasses.dataclass(kw_only=True, frozen=True)
class MpiProessInfo:
    rank: int


class MpiQueuedExecutorError(Exception):
    pass


type MpiMsg = MpiMsgExitOk | MpiMsgTerminate | MpiMsgJobParams | MpiMsgWorkerProcessReturns


class MpiMsgExitOk(t.NamedTuple):
    pass


class MpiMsgTerminate(t.NamedTuple):
    pass


class MpiMsgJobParams[*Args, Ret](t.NamedTuple):
    job_idx: int
    worker_process: t.Callable[[*Args], Ret]
    args: tuple[*Args]


class MpiMsgWorkerProcessReturns[*Args, Ret](t.NamedTuple):
    job_idx: int
    worker_process: t.Callable[[*Args], Ret]
    args: tuple[*Args]
    retval: Ret

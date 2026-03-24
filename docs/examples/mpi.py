import numpy as np
import time
import sorts
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sleep_sigma = 0.1


# `ParallelizableStep1` is a minimal `MpiQueuedExecution` implementation
class ParallelizableStep1(sorts.MpiQueuedExecution):

    def master_process(self):
        np.random.seed(123)
        worker_job_params_ls = [{"id": i} for i in range(20)]
        self.mpi_master_proc_loop(worker_job_params_ls)

    def worker_process(self, worker_job_params):
        print(f"{self.__class__.__name__} {rank=}, task id=", worker_job_params["id"])


# `ParallelizableStep2` adds sleep in `worker_process` to demonstrate the parallel execution more clearly
class ParallelizableStep2(sorts.MpiQueuedExecution):

    def master_process(self):
        np.random.seed(123)
        worker_job_params = [
            {"id": ind, "sleep": np.abs(np.random.randn()) * sleep_sigma} for ind in range(20)
        ]
        self.mpi_master_proc_loop(worker_job_params)

    def worker_process(self, worker_job_params):
        time.sleep(worker_job_params["sleep"])
        print(f"{self.__class__.__name__} {rank=}, task id=", worker_job_params["id"])


# to leverage mpi,
# - run this file with `mpiexec` e.g, `mpiexec -n 7 python <path_to_this_file>`
# - set `is_run_with_mpi=True`
# - set `progress=True` to enable terminal progress bar
res2 = ParallelizableStep1(is_run_with_mpi=True).run()
res1 = ParallelizableStep2(is_run_with_mpi=True, progress=True).run()

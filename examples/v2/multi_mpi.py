import numpy as np
import time
import sorts
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sleep_sigma = 0.1


class MpiBase(sorts.MpiQueuedExecution):

    def master_process(self):
        np.random.seed(123)
        tasks = [{"id": ind, "sleep": np.abs(np.random.randn()) * sleep_sigma} for ind in range(20)]
        self.mpi_master_proc_loop(tasks)

    def worker_process(self, task):
        time.sleep(task["sleep"])
        print(f"{self.__class__.__name__} {rank=}, task id=", task["id"])
        self.comm.send(True, dest=self.master_proc_rank)


class Mpi1(MpiBase):
    pass


class Mpi2(MpiBase):
    pass


res1 = Mpi1(is_run_with_mpi=True, progress=True).run()
res2 = Mpi2(is_run_with_mpi=True).run()

import numpy as np
import time
import sorts


class Script(sorts.MpiJobQueueExecutor):
    sleep_sigma = 0.1

    def master_main(self):
        np.random.seed(123)

        params_list = [(ind, np.abs(np.random.randn()) * self.sleep_sigma) for ind in range(20)]

        result = self.run_job_queue(
            worker_process=self.plus_one_and_sleep,
            job_params_list=params_list,
        )

    def plus_one_and_sleep(self, id, sleep_seconds):
        time.sleep(sleep_seconds)
        print(f"task id=", id)


try:
    from mpi4py import MPI

    pool_size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    pool_size = 1
    rank = 0

Script(is_run_with_mpi=pool_size > 1).run()

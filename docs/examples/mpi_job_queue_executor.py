import numpy as np
import time
import sorts


def gen_job_params():
    np.random.seed(123)
    sleep_sigma = 0.1

    worker_job_params = [
        {"id": ind, "sleep": np.abs(np.random.randn()) * sleep_sigma} for ind in range(20)
    ]
    return worker_job_params


def worker_process(params):
    params = sorts.utils.as_item_of_seq_retval(gen_job_params, params)
    time.sleep(params["sleep"])
    print(f"task id=", params["id"])


# mpi_executor = sorts.MpiQueuedExecutor(num_workers=7, is_run_with_mpi=True)
mpi_executor = sorts.MpiJobQueueExecutor(num_workers=7, is_run_with_mpi=False)

mpi_executor.run_job_queue(
    job_params_list=gen_job_params(),
    worker_process=worker_process,
    show_progress_bar=True,
)

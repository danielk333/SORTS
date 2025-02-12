"""Sets up a logging framework that can be imported and used anywhere.

"""
import json
import logging
import datetime
import pathlib
import time
import tracemalloc

import numpy as np
from tabulate import tabulate


class Profiler:
    """Performance profiler class.

    The named performance control is especially useful for timing contents of loops.

    Usage example:

        .. code-block:: python

            p = Profiler()
            p.start('list init')
            p.start('program')
            lst = list(range(200))
            p.stop('list init')
            for i in range(1000):
                p.start('list reversal')
                lst = lst[::-1]
                p.stop('list reversal')
            p.stop('program')

            print(p)

    """

    def __init__(self, distribution=False, track_memory=False, snapshot_total=True):
        self.distribution = distribution
        self.snapshot_total = snapshot_total
        self.track_memory = track_memory

        if self.track_memory:
            tracemalloc.start()

        self.exec_times = dict()
        self.start_times = dict()

        self.memory_stats = dict()
        self.snapshots = dict()

    def __del__(self):
        if self.track_memory:
            tracemalloc.stop()

    def snapshot(self, name):
        """Take a tracemalloc snapshot.

        :param str name: Name of the profiling location.
        """
        if not self.track_memory:
            return

        if self.snapshot_total:
            snap_ = tracemalloc.take_snapshot()
            alloc = snap_.statistics("lineno")
            size = sum([x.size for x in alloc])
            self.snapshots[name] = size
        else:
            self.snapshots[name] = tracemalloc.take_snapshot()

    def memory_diff(self, name, save=None):
        """Calculate a memory difference between the latest snapshot and now.

        :param str name: Name of the profiling location.
        :param str save: Save memory difference to this name, default is same as :code:`name`.
        """
        if not self.track_memory:
            return

        if save is None:
            save = name

        new_snapshot = tracemalloc.take_snapshot()

        if self.snapshot_total:
            new_alloc = new_snapshot.statistics("lineno")
            new_size = sum([x.size for x in new_alloc])

            data = new_size - self.snapshots[name]
        else:
            data = new_snapshot.compare_to(self.snapshots[name], "lineno")
            data.sort(key=lambda x: x.size_diff, reverse=True)

        if save in self.memory_stats:
            self.memory_stats[save].append(data)
        else:
            self.memory_stats[save] = [data]

    def start(self, name):
        """Records a start time for named call.

        :param str name: Name of the profiling location.
        """
        self.start_times[name] = time.time()

    def stop(self, name):
        """Records a time elapsed since the latest start time for named call.

        :param str name: Name of the profiling location.
        """
        dt = time.time() - self.start_times[name]
        self.start_times[name] = None

        if self.distribution:
            if name in self.exec_times:
                self.exec_times[name].append(dt)
            else:
                self.exec_times[name] = [dt]
        else:
            if name in self.exec_times:
                self.exec_times[name]["spent"] += dt
                self.exec_times[name]["num"] += 1
            else:
                self.exec_times[name] = {"spent": dt, "num": 1}

    def mean(self, name=None):
        if name is not None:
            if self.distribution:
                return np.mean(np.array(self.exec_times[name]))
            else:
                return self.exec_times[name]["spent"] / self.exec_times[name]["num"]
        else:
            ret = dict()
            for key in self.exec_times:
                if self.distribution:
                    ret[key] = np.mean(np.array(self.exec_times[key]))
                else:
                    ret[key] = self.exec_times[key]["spent"] / self.exec_times[key]["num"]
            return ret

    def total(self, name=None):
        if name is not None:
            if self.distribution:
                return np.sum(np.array(self.exec_times[name]))
            else:
                return self.exec_times[name]["spent"]
        else:
            ret = dict()
            for key in self.exec_times:
                if self.distribution:
                    ret[key] = np.sum(np.array(self.exec_times[key]))
                else:
                    ret[key] = self.exec_times[key]["spent"]
            return ret

    def fmt(self, normalize=None, timedelta=False):
        """Format summary of the profiler into a string.

        :param str normalize: Name of the profiling location to normalize execution time towards.
        :param bool timedelta: Print execution times as time-deltas.
        """
        means = self.mean()
        totals = self.total()

        data = []

        for key in self.exec_times:
            if self.distribution:
                num = str(len(self.exec_times[key]))
            else:
                num = str(self.exec_times[key]["num"])

            if timedelta:
                mu = str(datetime.timedelta(seconds=means[key]))
            else:
                mu = f"{means[key]:.5e} s"

            if normalize is not None:
                su = f"{totals[key]/totals[normalize]*100.0:.2f} %"
            else:
                if timedelta:
                    su = str(datetime.timedelta(seconds=totals[key]))
                else:
                    su = f"{totals[key]:.5e} s"

            data.append([key, num, mu, su])

        header = ["Name", "Executions", "Mean time", "Total time"]
        tab = str(tabulate(data, header, tablefmt="presto"))

        width = tab.find("\n", 0)

        str_ = f'{" Performance analysis ".center(width, "-")}\n'
        str_ += tab + "\n"
        str_ += "-" * width + "\n"

        if self.track_memory and self.snapshot_total:
            data = []

            for key in self.memory_stats:
                if len(self.memory_stats[key]) == 0:
                    continue
                else:
                    sum_ = np.sum(self.memory_stats[key]) / 1024.0
                    mean_ = sum_ / len(self.memory_stats[key])

                    su = f"{sum_:.5e} kB"
                    mu = f"{mean_:.5e} kB"

                data.append([key, len(self.memory_stats[key]), mu, su])

            header = ["Name", "Executions", "Mean size change", "Total size change"]
            tab = str(tabulate(data, header, tablefmt="presto"))

            width = tab.find("\n", 0)

            str_ += "\n" * 2 + f'{" Memory analysis ".center(width, "-")}\n'
            str_ += tab + "\n"
            str_ += "-" * width + "\n"

        return str_

    def __str__(self):
        return self.fmt()

    @classmethod
    def from_txt(cls, path):
        with open(path, "r") as f:
            dat = json.load(f)

        p = cls(distribution=dat["distribution"])
        p.exec_times = dat["exec_times"]
        return p

    def to_txt(self, path):
        dat = dict(
            distribution=self.distribution,
            exec_times=self.exec_times,
        )
        with open(path, "w") as f:
            json.dump(dat, f)


def add_logging_level(num, name):
    """Adds a custom logging level to all loggers.

    :param int num: Integer level for logging level.
    :param str name: Name of logging level.
    """

    def fn(self, message, *args, **kwargs):
        if self.isEnabledFor(num):
            self._log(num, message, args, **kwargs)

    logging.addLevelName(num, name)
    setattr(logging, name, num)
    return fn


logging.Logger.always = add_logging_level(100, "ALWAYS")


def _get_parallel():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        if comm.size > 1:
            parallel = comm.rank
        else:
            parallel = None
    except ImportError:
        parallel = None
    return parallel


def get_logger_formats():
    datefmt = "%Y-%m-%d %H:%M:%S"
    msecfmt = "%s.%03d"

    parallel = _get_parallel()

    if parallel is not None:
        format_str = f"%(asctime)s PID{parallel} %(levelname)-8s; %(message)s"
    else:
        format_str = "%(asctime)s %(levelname)-8s; %(message)s"

    return datefmt, msecfmt, format_str


def get_logger(
    name="sorts",
    path=None,
    file_level=logging.INFO,
    term_level=logging.INFO,
):
    """Creates a pre-configured logger. Formats to output both to terminal and a log file.

    **Note:** Can only use folders as paths for MPI usage.

    :param str name: Name of the logger.
    :param str/pathlib.Path path: Path to the logfile output. Can be a file, a folder or `None`.
    :param int file_level: Logging level of the file handler.
    :param int term_level: Logging level of the terminal handler.
    """
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_at_%H-%M")

    datefmt, msecfmt, format_str = get_logger_formats()
    parallel = _get_parallel()

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if path is not None:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if path.is_dir():
            if parallel is None:
                log_fname = path / f"{name}_{datetime_str}.log"
            else:
                log_fname = path / f"{name}_{datetime_str}_process{parallel}.log"
        else:
            if parallel is None:
                log_fname = str(path)
            else:
                new_name = path.name.replace(path.suffix, "") + f"_process{parallel}"
                log_fname = path.parent / f"{new_name}{path.suffix}"

        fh = logging.FileHandler(log_fname)
        fh.setLevel(file_level)  # debug and worse
        form_fh = logging.Formatter(format_str)
        form_fh.default_time_format = datefmt
        form_fh.default_msec_format = msecfmt
        fh.setFormatter(form_fh)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(term_level)
    form_ch = logging.Formatter(format_str)
    form_ch.default_time_format = datefmt
    form_ch.default_msec_format = msecfmt

    ch.setFormatter(form_ch)
    logger.addHandler(ch)

    return logger


def change_logfile(logger, path):
    """Deletes any previous `FileHandler` handlers and creates a new
    `FileHandler` to the new path with the same level as the previous one.

    :param logging.Logger logger: Logger object.
    :param str/pathlib.Path path: Path to the logfile output. Can be a file or folder.
    :param int file_level: Logging level of the file handler.
    :param int term_level: Logging level of the terminal handler.
    """
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_at_%H-%M")

    name = logger.name

    datefmt, msecfmt, format_str = get_logger_formats()
    parallel = _get_parallel()

    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.is_dir():
        if parallel is None:
            log_fname = path / f"{name}_{datetime_str}.log"
        else:
            log_fname = path / f"{name}_{datetime_str}_process{parallel}.log"
    else:
        if parallel is None:
            log_fname = path
        else:
            new_name = path.name.replace(path.suffix, "") + f"_process{parallel}"
            log_fname = path.parent / f"{new_name}{path.suffix}"

    fh = logging.FileHandler(log_fname)
    form_fh = logging.Formatter(format_str)
    form_fh.default_time_format = datefmt
    form_fh.default_msec_format = msecfmt
    fh.setFormatter(form_fh)

    for hdl in logger.handlers[:]:
        if isinstance(hdl, logging.FileHandler):
            fh.setLevel(hdl.level)
            logger.removeHandler(hdl)

    logger.addHandler(fh)

    return logger


def term_level(logger, level):
    """Set the StreamHandler level."""
    for hdl in logger.handlers[:]:
        if not isinstance(hdl, logging.StreamHandler):
            continue

        if isinstance(level, str):
            level = getattr(logging, level)

        hdl.setLevel(level)

    return logger


def file_level(logger, level):
    """Set the FileHandler level."""
    for hdl in logger.handlers[:]:
        if not isinstance(hdl, logging.FileHandler):
            continue

        if isinstance(level, str):
            level = getattr(logging, level)

        hdl.setLevel(level)

    return logger

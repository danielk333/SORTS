#!/usr/bin/env python

'''Main simulation handler in the form of a class using the capabilities of the entire toolbox.

'''

#Python standard import
import pathlib
import shutil
from collections import OrderedDict
import logging

#Third party import
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    MPI = None

#Local import
from . import profiling

def mpi_wrap_master_thread(func):
    def master_th_func(*args, **kwargs):
        if MPI is not None:
            if comm.rank == 0:
                ret = func(*args, **kwargs)
            else:
                ret = None
            comm.barrier()
        else:
            ret = func(*args, **kwargs)
        return ret
    return master_th_func


@mpi_wrap_master_thread
def mpi_mkdir(path):
    path.mkdir(exist_ok=True)


@mpi_wrap_master_thread
def mpi_rmdir(path):
    shutil.rmtree(path)


@mpi_wrap_master_thread
def mpi_copy(src, dst):
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)




class Simulation:
    '''Convenience simulation handler, creates a step-by-step simulation sequence and creates file system structure for saving of data to disk.
    '''
    def __init__(self, scheduler, root, paths=['logs'], logger=True, profiler=True, **kwargs):

        self.scheduler = scheduler
        if not isinstance(root, pathlib.Path):
            root = pathlib.Path(root)

        self.root = root
        self.paths = paths

        self.branch_name = 'master'

        if not self.root.is_dir():
            mpi_mkdir(self.root)

        _master = self.root / self.branch_name
        if not _master.is_dir():
            mpi_mkdir(_master)
            for path in self.paths:
                mpi_mkdir(self.root / self.branch_name / path)


        if logger:
            if 'logs' in self.paths:
                lpath = self.root / self.branch_name / 'logs'
            else:
                lpath = None

            self.logger = profiling.get_logger(
                'Simulation',
                path = lpath,
                file_level = kwargs.get('file_level', logging.INFO),
                term_level = kwargs.get('term_level', logging.INFO),
            )
            self.scheduler.logger = self.logger
        else:
            self.logger = None

        if profiler:
            self.profiler = profiling.Profiler()
            self.scheduler.profiler = self.profiler
        else:
            self.profiler = None

        self.steps = OrderedDict()


    def get_path(self, name=None):
        if name is None:
            return self.root / self.branch_name
        else:
            return self.root / self.branch_name / name


    def delete(self, branch):
        '''Delete branch.
        '''
        mpi_rmdir(self.root / branch)
        if self.branch_name == branch:
            mpi_mkdir(self.root / self.branch_name)
            for path in self.paths:
                mpi_mkdir(self.root / self.branch_name / path)

    def branch(self, name):
        '''Create branch.
        '''
        if (self.root / name).is_dir():
            raise ValueError(f'Branch "{name}" already exists')

        mpi_copy(self.root / self.branch_name, self.root / name)
        self.branch_name = name


    def checkout(self, branch):
        '''Change to branch.
        '''
        self.branch_name = branch


    def run(self, step = None):

        if step is None:
            for name, func in self.steps.items():
                if self.profiler is not None: self.profiler.start(f'Simulation:run:{name}')
                func()
                if self.profiler is not None: self.profiler.stop(f'Simulation:run:{name}')
        else:
            func = self.steps[step]
            if self.profiler is not None: self.profiler.start(f'Simulation:run:{step}')
            func()
            if self.profiler is not None: self.profiler.stop(f'Simulation:run:{step}')


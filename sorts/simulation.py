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
    comm = None

#Local import
from . import profiling

def mpi_wrap_master_thread(func):
    def master_th_func(*args, **kwargs):
        if comm is not None:
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



def simulation_step(attribute=None, MPI=True, h5_cache=True):

    def step_wrapping(func):

        def wrapped_step(self, *args, **kwargs):

            if attribute is not None:
                attr = getattr(self, attribute)
                if MPI and comm is not None:
                    _iter = range(attr.rank, len(attr), comm.size)
                else:
                    _iter = range(len(attr))

                for index, item in enumerate(attr):
                    ret = func(self, index, item, *args, **kwargs)
            else:
                ret = func(self, *args, **kwargs)
                



        for ind in range(len(self.objs)):
            fname = self.get_path('objs') / f'obj{ind}.h5'

            if fname.is_file():
                with h5py.File(fname,'r') as h:
                    t = h['t'][()]
                    state = h['state'][()]
            else:
                t = sorts.equidistant_sampling(
                    orbit = self.objs[ind].orbit, 
                    start_t = self.scheduler.controllers[0].t.min(), 
                    end_t = self.scheduler.controllers[0].t.max(), 
                    max_dpos=1e3,
                )
                state = self.objs[ind].get_state(t)
                with h5py.File(fname,'w') as h:
                    h.create_dataset('t', data=t)
                    h.create_dataset('state', data=state)

            self.states[ind] = state
            self.ts[ind] = t



class Simulation:
    '''Convenience simulation handler, creates a step-by-step simulation sequence and creates file system structure for saving of data to disk.
    '''
    def __init__(self, scheduler, root, logger=True, profiler=True, **kwargs):

        self.scheduler = scheduler
        if not isinstance(root, pathlib.Path):
            root = pathlib.Path(root)

        self.root = root

        self.branch_name = 'master'

        if not self.root.is_dir():
            mpi_mkdir(self.root)

        _master = self.root / self.branch_name
        if not _master.is_dir():
            mpi_mkdir(_master)


        if logger:
            self.logger = profiling.get_logger(
                'Simulation',
                path = self.log_path,
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


    def make_paths(self):
        for path in self.paths:
            mpi_mkdir(self.get_path(path))


    @property
    def paths(self):
        return [key for key in self.steps]


    @property
    def log_path(self):
        return self.root / self.branch_name / 'logs'



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
            self.make_paths()


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
        self.logger = profiling.change_logfile(self.logger, self.log_path)


    def run(self, step = None):

        self.make_paths()

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


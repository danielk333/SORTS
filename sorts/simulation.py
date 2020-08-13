#!/usr/bin/env python

'''Main simulation handler in the form of a class using the capabilities of the entire toolbox.

'''

#Python standard import
import pathlib
import shutil
from collections import OrderedDict
import logging
import copy
import pickle

#Third party import
import h5py
import numpy as np

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



def simulation_step(iterable=None, store=None, MPI=False, h5_cache=False, pickle_cache=False, MPI_mode=None):
    '''Simulation step decorator

    :param str MPI_mode: Mode of operations on node-data communication, available options are "gather", None, "allgather" and "barrier".

    '''

    def step_wrapping(func):

        def wrapped_step(self, step, *args, **kwargs):

            if h5_cache or pickle_cache:
                dir_ = self.get_path(step)
                if not dir_.is_dir():
                    mpi_mkdir(dir_)

            if iterable is not None:
                attr = getattr(self, iterable)
                if MPI and comm is not None:
                    _iter = list(range(comm.rank, len(attr), comm.size))
                else:
                    _iter = list(range(len(attr)))

                rets = [None]*len(attr)

                for index in _iter:
                    item = attr[index]
                    if h5_cache:
                        fname = dir_ / f'{iterable}_{index}.h5'
                        if fname.is_file():
                            with h5py.File(fname,'r') as h:
                                ret = {}
                                for key in h:
                                    ret[key] = h[key][()].copy()
                                for key in h.attrs:
                                    ret[key] = copy.copy(h.attrs[key])
                        else:
                            ret = func(self, index, item, *args, **kwargs)
                            if ret is None: ret = {}

                            with h5py.File(fname,'w') as h:
                                for key in ret:
                                    if isinstance(ret[key], np.ndarray):
                                        h.create_dataset(key, data=ret[key])
                                    else:
                                        h.attrs[key] = ret[key]
                    elif pickle_cache:
                        fname = dir_ / f'{iterable}_{index}.pickle'
                        if fname.is_file():
                            with open(fname, 'rb') as h:
                                ret = pickle.load(h)
                        else:
                            ret = func(self, index, item, *args, **kwargs)
                            if ret is None: ret = {}

                            with open(fname, 'wb') as h:
                                pickle.dump(ret, h)

                    else:
                        ret = func(self, index, item, *args, **kwargs)
                        if ret is None: ret = {}

                    rets[index] = ret

                if MPI and comm is not None:
                    mpi_inds = []
                    for thrid in range(comm.size):
                        mpi_inds.append(range(thrid, len(attr), comm.size))

                    if MPI_mode is None:
                        pass
                    elif MPI_mode == 'barrier':
                        comm.barrier()
                    elif MPI_mode == 'gather':

                        if comm.rank == 0:
                            for thr_id in range(comm.size):
                                if thr_id != 0:
                                    for ind in mpi_inds[thr_id]:
                                        rets[ind] = comm.recv(source=thr_id, tag=ind)

                        else:
                            for ind in mpi_inds[comm.rank]:
                                comm.send(rets[ind], dest=0, tag=ind)
                        
                        #delete data from threads to save memory
                        if comm.rank != 0:
                            for ind in mpi_inds[comm.rank]:
                                rets[ind] = None

                    elif MPI_mode == 'allgather':

                        for thr_id in range(comm.size):
                            for ind in mpi_inds[thr_id]:
                                rets[ind] = comm.bcast(rets[ind], root=thr_id)

                else:
                    raise ValueError(f'MPI_mode "{MPI_mode}" not valid')

            else:
                if h5_cache:
                    fname = dir_ / f'data.h5'
                    if fname.is_file():
                        with h5py.File(fname,'r') as h:
                            rets = {}
                            for key in h:
                                rets[key] = h[key][()].copy()
                            for key in h.attrs:
                                rets[key] = copy.copy(h.attrs[key])
                    else:
                        rets = func(self, *args, **kwargs)
                        if rets is None: rets = {}

                        with h5py.File(fname,'w') as h:
                            for key in rets:
                                if isinstance(rets[key], np.ndarray):
                                    h.create_dataset(key, data=rets[key])
                                else:
                                    h.attrs[key] = rets[key]
                elif pickle_cache:
                    fname = dir_ / f'data.pickle'
                    if fname.is_file():
                        with open(fname, 'rb') as h:
                            rets = pickle.load(h)
                    else:
                        rets = func(self, *args, **kwargs)
                        if rets is None: rets = {}

                        with open(fname, 'wb') as h:
                            pickle.dump(rets, h)
                else:
                    rets = func(self, *args, **kwargs)
                    if rets is None: rets = {}
                

            if store is not None:
                setattr(self, store, rets)

            return rets

        wrapped_step._simulation_step = True

        return wrapped_step

    return step_wrapping




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


    def run(self, step = None, *args, **kwargs):

        self.make_paths()

        if step is None:
            for name, func in self.steps.items():
                if self.profiler is not None: self.profiler.start(f'Simulation:run:{name}')
                if hasattr(func, '_simulation_step'):
                    func(name, *args, **kwargs)
                else:
                    func(*args, **kwargs)
                if self.profiler is not None: self.profiler.stop(f'Simulation:run:{name}')
        else:
            func = self.steps[step]
            if self.profiler is not None: self.profiler.start(f'Simulation:run:{step}')
            if hasattr(func, '_simulation_step'):
                func(step, *args, **kwargs)
            else:
                func(*args, **kwargs)
            if self.profiler is not None: self.profiler.stop(f'Simulation:run:{step}')


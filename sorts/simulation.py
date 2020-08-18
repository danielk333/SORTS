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


def MPI_single_process(process_id):

    def step_wrapping(func):

        def wrapped_step(self, *args, **kwargs):
            if comm is not None:
                if comm.rank == process_id:
                    rets = func(self, *args, **kwargs)
                else:
                    rets = None
            else:
                rets = func(self, *args, **kwargs)
            return rets
        return wrapped_step

    return step_wrapping


def simulation_step(iterable=None, store=None, MPI=False, caches=[], post_MPI=None, MPI_only=None):
    '''Simulation step decorator

 
    :param str caches: Is a list of strings for the caches to be used, available by default is "h5" and "pickle". Custom caches are implemented by implementing methods with the string name but prefixed with load_ and save_.
    :param str post_MPI: Mode of operations on node-data communication, available options are None, "gather", "allbcast" and "barrier".

    '''

    if isinstance(caches, str):
        caches = [caches]
    if isinstance(iterable, str):
        iterable = [iterable]
    if isinstance(store, str):
        store = [store]

    def step_wrapping(func):

        def wrapped_step(self, step, *args, **kwargs):

            if len(caches) > 0:
                dir_ = self.get_path(step)
                if not dir_.is_dir():
                    mpi_mkdir(dir_)
            else:
                dir_ = None

            if iterable is not None:
                all_attrs = []
                for var in iterable:
                    subvars = var.split('.')
                    for vari, subvar in enumerate(subvars):
                        if vari == 0:
                            obj = getattr(self, subvar)
                        else:
                            obj = getattr(obj, subvar)
                    all_attrs.append(obj)

                if len(all_attrs) == 1:
                    attr = all_attrs[0]
                else:
                    attr = list(zip(*all_attrs))


                if MPI and comm is not None and MPI_only is None:
                    _iter = list(range(comm.rank, len(attr), comm.size))
                else:
                    _iter = list(range(len(attr)))

                rets = [None]*len(attr)

                for index in _iter:
                    if comm is not None and MPI_only is not None:
                        if comm.rank != MPI_only:
                            continue
                    item = attr[index]
                    loaded_ = False

                    #load
                    for cache in caches:
                        fname = dir_ / f'{"_".join(iterable)}_{index}.{cache}'
                        if fname.is_file():
                            lfunc = getattr(self, f'load_{cache}')
                            ret = lfunc(fname)
                            loaded_ = True
                            break

                    #if there are no caches
                    if not loaded_:
                        ret = func(self, index, item, *args, **kwargs)

                        #save
                        for cache in caches:
                            fname = dir_ / f'{"_".join(iterable)}_{index}.{cache}'
                            sfunc = getattr(self, f'save_{cache}')
                            sfunc(fname, ret)

                    rets[index] = ret

                if MPI and comm is not None:
                    mpi_inds = []
                    for thrid in range(comm.size):
                        mpi_inds.append(range(thrid, len(attr), comm.size))

                    if post_MPI is None:
                        pass
                    elif post_MPI == 'barrier':
                        comm.barrier()
                    elif post_MPI.startswith('gather'):

                        if comm.rank == 0:
                            for thr_id in range(comm.size):
                                if thr_id != 0:
                                    for ind in mpi_inds[thr_id]:
                                        rets[ind] = comm.recv(source=thr_id, tag=ind)

                        else:
                            for ind in mpi_inds[comm.rank]:
                                comm.send(rets[ind], dest=0, tag=ind)
                        
                        if post_MPI == 'gather-clear':
                            if comm.rank != 0:
                                for ind in mpi_inds[comm.rank]:
                                    rets[ind] = None

                    elif post_MPI == 'allbcast':

                        for thr_id in range(comm.size):
                            for ind in mpi_inds[thr_id]:
                                rets[ind] = comm.bcast(rets[ind], root=thr_id)

                    elif post_MPI == 'bcast' and MPI_only is not None:
                        for ind in _iter:
                            rets[ind] = comm.bcast(rets[ind], root=MPI_only)

                else:
                    raise ValueError(f'post_MPI "{post_MPI}" not valid')

            else:
                loaded_ = False
                rets = None

                do_func = True
                if comm is not None and MPI_only is not None:
                    if comm.rank != MPI_only:
                        do_func = False

                if do_func:
                    #load
                    for cache in caches:
                        fname = dir_ / f'{step}.{cache}'
                        if fname.is_file():
                            lfunc = getattr(self, f'load_{cache}')
                            rets = lfunc(fname)
                            loaded_ = True
                            break

                    #if there are no caches
                    if not loaded_:
                        rets = func(self, *args, **kwargs)

                        #save
                        for cache in caches:
                            fname = dir_ / f'{step}.{cache}'
                            sfunc = getattr(self, f'save_{cache}')
                            sfunc(fname, rets)

                if post_MPI is None:
                    pass
                elif post_MPI == 'bcast' and MPI_only is not None:
                    rets = comm.bcast(rets, root=MPI_only)

            if store is not None:
                for si, var in enumerate(store):
                    subvars = var.split('.')
                    if len(subvars) == 1:
                        obj = self
                        name = subvars[0]
                    else:
                        for vari, subvar in enumerate(subvars[:-1]):
                            if vari == 0:
                                obj = getattr(self, subvar)
                            else:
                                obj = getattr(obj, subvar)
                        name = subvars[-1]
                    
                    if iterable is not None:
                        iter_obj = [None]*len(attr)
                        for index in range(len(attr)):
                            if len(store) == 1 or rets[index] is None:
                                iter_obj[index] = rets[index]
                            else:
                                iter_obj[index] = rets[index][si]
                        setattr(obj, name, iter_obj)
                    else:
                        if len(store) == 1 or rets is None:
                            setattr(obj, name, rets)
                        else:
                            setattr(obj, name, rets[si])

            return rets

        wrapped_step._simulation_step = True

        return wrapped_step

    return step_wrapping




class Simulation:
    '''Convenience simulation handler, creates a step-by-step simulation sequence and creates file system structure for saving of data to disk.
    '''
    def __init__(self, scheduler, root, logger=True, profiler=True, **kwargs):
        self.steps = OrderedDict()
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

        self.make_paths()

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


    def save_pickle(self, path, data):
        with open(path, 'wb') as h:
            pickle.dump(data, h)

    def load_pickle(self, path):
        if path.is_file():
            with open(path, 'rb') as h:
                ret = pickle.load(h)
            return ret

    def save_h5(self, path, data):
        with h5py.File(path,'w') as h:
            if isinstance(data, dict):
                for key in data:
                    if isinstance(data[key], np.ndarray):
                        h.create_dataset(key, data=data[key])
                    else:
                        h.attrs[key] = data[key]
            else:
                h.create_dataset('saved_data__', data=data)

    def load_h5(self, path):
        if path.is_file():
            with h5py.File(path,'r') as h:
                if 'saved_data__' in h:
                    ret = h['saved_data__'][()].copy()
                else:
                    ret = {}
                    for key in h:
                        ret[key] = h[key][()].copy()
                    for key in h.attrs:
                        ret[key] = copy.copy(h.attrs[key])
            return ret


    @staticmethod
    def MPI_bcast(process_id, data):
        if comm is not None:
            data = comm.bcast(data, root=process_id)
        return data


    @staticmethod
    def MPI_barrier():
        if comm is not None:
            comm.barrier()

    


    def make_paths(self):
        for path in self.paths:
            mpi_mkdir(self.get_path(path))


    @property
    def paths(self):
        return [key for key in self.steps] + ['logs']


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
        if (self.root / branch).is_dir():
            mpi_rmdir(self.root / branch)
        if self.branch_name == branch:
            mpi_mkdir(self.root / self.branch_name)
            self.make_paths()


    def branch(self, name, empty=False):
        '''Create branch.
        '''
        if (self.root / name).is_dir():
            raise ValueError(f'Branch "{name}" already exists')

        if empty:
            mpi_mkdir(self.root / name)
            for path in self.paths:
                mpi_mkdir(self.root / name / path)
        else:
            mpi_copy(self.root / self.branch_name, self.root / name)
            self.checkout(name)

        self.checkout(name)



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


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
import datetime
from glob import glob

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


def log_exceptions(func):

    def wrapped_step(self, *args, **kwargs):

        try:
            rets = func(self, *args, **kwargs)
        except BaseException as err:
            if hasattr(self, 'logger'):
                if self.logger is not None:
                    self.logger.exception(f'\nargs: {args}\n kwargs: {kwargs}')
            raise err
            
        return rets

    return wrapped_step




def MPI_single_process(process_id):
    '''Simulation step single process method restriction decorator

    '''
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

        if hasattr(func, '_simulation_step'):
            wrapped_step._simulation_step = func._simulation_step
        return wrapped_step

    return step_wrapping


def MPI_action(action, iterable = False, root = 0):
    '''Simulation step MPI post step action decorator

    :param str action: Mode of operations on node-data communication, available options are "gather", "gather-clear", "bcast" and "barrier".
    '''

    def step_wrapping(func):

        def wrapped_step(self, *args, **kwargs):
            rets = func(self, *args, **kwargs)

            if comm is None:
                return rets

            if iterable:
                mpi_inds = []
                for thrid in range(comm.size):
                    mpi_inds.append(range(thrid, len(rets), comm.size))


            if action == 'barrier':
                comm.barrier()
            elif action.startswith('gather'):
                if iterable:
                    if comm.rank == root:
                        for thr_id in range(comm.size):
                            if thr_id != root:
                                for ind in mpi_inds[thr_id]:
                                    rets[ind] = comm.recv(source=thr_id, tag=ind)

                    else:
                        for ind in mpi_inds[comm.rank]:
                            comm.send(rets[ind], dest=root, tag=ind)
                    
                    if action == 'gather-clear':
                        if comm.rank != root:
                            for ind in mpi_inds[comm.rank]:
                                rets[ind] = None
                else:
                    all_rets = [None]*comm.size
                    all_rets[comm.rank] = rets

                    if comm.rank == root:
                        for thr_id in range(comm.size):
                            if thr_id != root:
                                all_rets[thr_id] = comm.recv(source=thr_id, tag=thr_id)
                    else:
                        comm.send(all_rets[comm.rank], dest=root, tag=comm.rank)
                    
                    if action == 'gather-clear':
                        if comm.rank != root:
                            all_rets[comm.rank] = None
                    rets = all_rets

            elif action == 'bcast':

                if iterable:
                    for thr_id in range(comm.size):
                        for ind in mpi_inds[thr_id]:
                            rets[ind] = comm.bcast(rets[ind], root=thr_id)
                else:     
                    rets = comm.bcast(rets, root=root)

            return rets

        if hasattr(func, '_simulation_step'):
            wrapped_step._simulation_step = func._simulation_step
        return wrapped_step

    return step_wrapping


def iterable_step(iterable, MPI=False, log=False, reduce=None):
    '''Simulation step iteration decorator

 
    '''
    reduce_ = reduce;

    if isinstance(iterable, str):
        iterable = [iterable]

    def step_wrapping(func):

        def wrapped_step(self, *args, **kwargs):

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


            if MPI and comm is not None:
                _iter = list(range(comm.rank, len(attr), comm.size))
            else:
                _iter = list(range(len(attr)))

            if reduce_ is None:
                rets = [None]*len(attr)
            else:
                rets = None


            step_name = kwargs.get('_step_name', None)

            profiler_name = f'Simulation:iterable_step_{step_name}'

            _iters = 0
            _total = len(_iter)

            for index in _iter:
                if log and self.profiler is not None:
                    self.profiler.start(profiler_name)
                item = attr[index]

                if hasattr(func, '_simulation_step'):
                    kwargs['_iterable_index'] = index

                ret = func(self, index, item, *args, **kwargs)
                if reduce_ is None:
                    rets[index] = ret
                else:
                    rets = reduce_(rets, ret)
                    del ret

                if log and self.profiler is not None:
                    self.profiler.stop(profiler_name)
                _iters += 1
                if log and self.logger is not None:
                    if self.profiler is not None:
                        _spent = self.profiler.total(name=profiler_name)
                        _est_left = (_total - _iters)*self.profiler.mean(name=profiler_name)
                        self.logger.always(f'Simulation:{step_name}:iterable_step: {_iters}/{_total}\n'
                            + f'[Elapsed  ] {str(datetime.timedelta(seconds=_spent))} | '
                            + f'[Time left] {str(datetime.timedelta(seconds=_est_left))}'
                        )
                    else:
                        self.logger.always(f'Simulation:{step_name}:iterable_step: {_iters}/{_total}')

            if log and self.profiler is not None:
                if step_name is None:
                    del self.profiler.exec_times[profiler_name]

            return rets

        wrapped_step._simulation_step = True
        return wrapped_step

    return step_wrapping



def store_step(store, iterable=False):
    '''Simulation step storing decorator

 
    '''
    if isinstance(store, str):
        store = [store]

    def step_wrapping(func):

        def wrapped_step(self, *args, **kwargs):

            rets = func(self, *args, **kwargs)

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
                
                if iterable:
                    iter_obj = [None]*len(rets)
                    for index in range(len(rets)):
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



def iterable_cache(steps, caches, MPI=False, log=False, reduce=None):
    '''Simulation step iteration decorator

 
    '''
    reduce_ = reduce;

    if isinstance(steps, str):
        steps = [steps]
    if isinstance(caches, str):
        caches = [caches]


    def step_wrapping(func):

        def wrapped_step(self, *args, **kwargs):

            all_files = []
            all_indecies = []
            for step, cache in zip(steps, caches):
                dir_ = self.get_path(step)
                if not dir_.is_dir():
                    raise ValueError(f'Input step {step} has no cache {cache}')
                files = [pathlib.Path(x) for x in glob(str(dir_ / f'*.{cache}'))]
                indecies = [int(file.stem.split('_')[0]) for file in files]
                
                files = [x for _, x in sorted(zip(indecies,files), key=lambda pair: pair[0])]
                indecies = sorted(indecies)

                all_files += [files]
                all_indecies += [indecies]

            if len(steps) == 1:
                files_lst = [[x] for x in all_files[0]]
            else:
                #check all inds exist
                for index, inds in enumerate(zip(*all_indecies)):
                    if len(set(inds)) > 1:
                        raise ValueError(f'Missing index {inds}')
                files_lst = list(zip(*all_files))
            indecies = all_indecies[0]

            if MPI and comm is not None:
                _iter = list(range(comm.rank, len(indecies), comm.size))
            else:
                _iter = list(range(len(indecies)))

            if reduce_ is None:
                rets = [None]*len(indecies)
            else:
                rets = None

            step_name = kwargs.get('_step_name', None)

            profiler_name = f'Simulation:iterable_cache:{step_name}'

            _iters = 0
            _total = len(_iter)

            for index__ in _iter:
                index = indecies[index__]
                files = files_lst[index__]

                if log and self.profiler is not None:
                    self.profiler.start(profiler_name)
                
                item = []
                for cache, fname in zip(caches, files):
                    lfunc = getattr(self, f'load_{cache}')
                    item += [lfunc(fname)]

                if len(caches) == 1:
                    item = item[0]

                if hasattr(func, '_simulation_step'):
                    kwargs['_iterable_index'] = index

                ret = func(self, index, item, *args, **kwargs)
                if reduce_ is None:
                    rets[index__] = ret
                else:
                    rets = reduce_(rets, ret)
                    del ret

                if log and self.profiler is not None:
                    self.profiler.stop(profiler_name)
                _iters += 1
                if log and self.logger is not None:
                    if self.profiler is not None:
                        _spent = self.profiler.total(name=profiler_name)
                        _est_left = (_total - _iters)*self.profiler.mean(name=profiler_name)
                        self.logger.always(f'Simulation:{step_name}:iterable_cache: {_iters}/{_total}\n'
                            + f'[Elapsed  ] {str(datetime.timedelta(seconds=_spent))} | '
                            + f'[Time left] {str(datetime.timedelta(seconds=_est_left))}'
                        )
                    else:
                        self.logger.always(f'Simulation:{step_name}:iterable_cache: {_iters}/{_total}')

            if log and self.profiler is not None:
                if step_name is None:
                    del self.profiler.exec_times[profiler_name]

            return rets


        wrapped_step._simulation_step = True
        return wrapped_step

    return step_wrapping




def cached_step(caches):
    '''Simulation step caching decorator

 
    :param str caches: Is a list of strings for the caches to be used, available by default is "h5" and "pickle". Custom caches are implemented by implementing methods with the string name but prefixed with load_ and save_.

    '''
    if isinstance(caches, str):
        caches = [caches]

    def step_wrapping(func):

        def wrapped_step(self, *args, **kwargs):

            step = kwargs.get('_step_name', 'cached_data')
            fname_parts = kwargs.pop('_fname_parts', ['data'])
            index = kwargs.get('_iterable_index', None)
            if index is None:
                index_lst = []
            else:
                index_lst = [str(index)]

            dir_ = self.get_path(step)
            if not dir_.is_dir():
                mpi_mkdir(dir_)

            loaded_ = False

            #load
            for cache in caches:
                fname = dir_ / f'{"_".join(index_lst + fname_parts)}.{cache}'
                if fname.is_file():
                    lfunc = getattr(self, f'load_{cache}')
                    try:
                        ret = lfunc(fname)
                        loaded_ = True
                    except (OSError, EOFError, UnicodeError, ):
                        fname.unlink()

                if loaded_:
                    break
                    

            #if there are no caches
            if not loaded_:
                ret = func(self, *args, **kwargs)

                #save
                for cache in caches:
                    fname = dir_ / f'{"_".join(index_lst + fname_parts)}.{cache}'
                    sfunc = getattr(self, f'save_{cache}')
                    sfunc(fname, ret)

            return ret

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
            if isinstance(data, list) or isinstance(data, tuple):
                for ind in range(len(data)):
                    h.attrs['__ret_len'] = len(data)
                    if isinstance(data[ind], np.ndarray):
                        h.create_dataset(f'saved_data__{ind}', data=data[ind])
                    else:
                        h.attrs[f'saved_data__{ind}'] = data[ind]
            else:
                h.create_dataset('__saved_data', data=data)

    def load_h5(self, path):
        if path.is_file():
            with h5py.File(path,'r') as h:
                if '__saved_data' in h:
                    ret = h['__saved_data'][()].copy()

                elif '__ret_len' in h.attrs:
                    ret = [None]*copy.copy(h.attrs['__ret_len'])
                    for ind in range(len(ret)):
                        key = f'saved_data__{ind}'
                        if key in h:
                            ret[ind] = h[key][()].copy()
                        else:
                            ret[ind] = copy.copy(h.attrs[key])

                else:
                    ret = {}
                    for key in h:
                        ret[key] = h[key][()].copy()
                    for key in h.attrs:
                        ret[key] = copy.copy(h.attrs[key])
            return ret




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
            if self.logger is not None:
                self.logger.info(f'Branch "{name}" already exists')
                return
        else:
            if empty:
                mpi_mkdir(self.root / name)
                for path in self.paths:
                    mpi_mkdir(self.root / name / path)
            else:
                mpi_copy(self.root / self.branch_name, self.root / name)

        self.checkout(name)



    def checkout(self, branch):
        '''Change to branch.
        '''
        self.branch_name = branch
        self.logger = profiling.change_logfile(self.logger, self.log_path)


    @log_exceptions
    def run(self, step = None, *args, **kwargs):

        self.make_paths()

        if step is None:
            for name, func in self.steps.items():
                if self.profiler is not None: self.profiler.start(f'Simulation:run:{name}')
                if self.logger is not None: self.logger.info(f'Simulation:run:{name}')

                if hasattr(func, '_simulation_step'):
                    func(*args, _step_name=name, **kwargs)
                else:
                    func(*args, **kwargs)
                if self.profiler is not None: self.profiler.stop(f'Simulation:run:{name}')
                if self.logger is not None: self.logger.info(f'Simulation:run:{name} [completed]')
        else:
            func = self.steps[step]
            if self.profiler is not None: self.profiler.start(f'Simulation:run:{step}')
            if self.logger is not None: self.logger.info(f'Simulation:run:{step}')
            if hasattr(func, '_simulation_step'):
                func(*args, _step_name=step, **kwargs)
            else:
                func(*args, **kwargs)
            if self.profiler is not None: self.profiler.stop(f'Simulation:run:{step}')
            if self.logger is not None: self.logger.info(f'Simulation:run:{step} [completed]')


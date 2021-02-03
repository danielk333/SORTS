#!/usr/bin/env python

'''Contains all helper functions to automate parallelization with MPI, handle caching and stepping of simulations.

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
import os
import argparse

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
    '''Wrap function to only execute on thread rank=0
    '''
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
    '''Make directory on thread rank=0
    '''
    path.mkdir(exist_ok=True)


@mpi_wrap_master_thread
def mpi_rmdir(path):
    '''Remove directory on thread rank=0 with :code:`shutil.rmtree`
    '''
    shutil.rmtree(path)


@mpi_wrap_master_thread
def mpi_copy(src, dst, linkfiles=False):
    '''Copy path on thread rank=0 with :code:`shutil.copytree` or :code:`copy2` for files. If :code:`linkfiles` is true, files are soft-linked rather then copied.
    '''
    if src.is_dir():
        if linkfiles:
            shutil.copytree(src, dst, copy_function=os.link)
        else:
            shutil.copytree(src, dst)
    else:
        if linkfiles:
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)


def log_exceptions(func):
    '''If instance has a logger, log exceptions raised by this method.
    '''
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
    '''Simulation step single process method restriction decorator.

    :param int process_id: The process id the wrapped function should only execute on. All other processes return :code:`None`.

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
    '''Simulation step MPI post step action decorator.

    :param str action: Mode of operations on node-data communication, available options are "gather", "gather-clear", "bcast" and "barrier".
    :param bool iterable: Indicates if the "gather", "gather-clear" or "bcast" should consider an iterable (that has been parallelized with MPI).
    :param int root: The target MPI process for the "gather", "gather-clear" and for the source process for "bcast" if :code:`iterable=False`.
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
    '''Simulation step iteration decorator.

    :param str/list iterable: The name/list of names of the instance properties (fetched using :code:`getattr`) to iterate over. Can be multiple levels, e.g. :code:`object.subobject.a_list`.
    :param bool MPI: Determines if the iteration should be MPI-parallelized
    :param bool log: Use the :code:`self.logger` instance, if it exists, to log the execution of the iteration.
    :param function reduce: A pointer to the binary-function used to reduce the results.
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

    :param str/list store: The name/list names of the properties to save the method return as (set using :code:`setattr`). Can be multiple levels, e.g. :code:`object.subobject.a_property`. The order of the names correspond to the order of method returned variables.
    :param bool iterable: Determines if the return of the method is an iteration or not. If its an iteration, it splits the return values into different lists based on the number of variables.
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
    '''Simulation step cache iteration decorator

    :param str/list steps: The name/list of names of the cached steps to be iterated over. It uses the step name to find the files in the corresponding folder.
    :param str/list caches: The name/list of cache-methods to be used to load the caches of the steps.
    :param bool MPI: Determines if the iteration should be MPI-parallelized.
    :param bool log: Use the :code:`self.logger` instance, if it exists, to log the execution of the iteration.
    :param function reduce: A pointer to the binary-function used to reduce the results.
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
                files = [pathlib.Path(x) for x in glob(str(dir_ / f'*'))]
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

 
    :param str caches: Is a list of strings for the caches to be used, available by default is "h5" and "pickle". Custom caches are implemented by implementing methods with the string name but prefixed with :code:`load_` and :code:`save_`.

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

    :param Scheduler scheduler: A scheduler instance to run. This input is used to assure that the same logger and profiler is used for the Simulation and the Scheduler.
    :param str/pathlib.Path root: The path to the root folder where all files will be stored.
    :param bool logger: If :code:`False`, do not instantiate a logger.
    :param bool profiler: If :code:`False`, do not instantiate a profiler.
    '''
    def __init__(self, scheduler, root, logger=True, profiler=True, **kwargs):
        self.steps = OrderedDict()
        self.scheduler = scheduler

        if root is None:
            self.persistancy = False
        else:
            self.persistancy = True

        self.root = root

        self.__user_args = []

        if self.persistancy:
            if not isinstance(self.root, pathlib.Path):
                self.root = pathlib.Path(self.root)

            if not self.root.is_dir():
                mpi_mkdir(self.root)


        self.branch_name = 'master'

        if self.persistancy:
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
            if self.scheduler is not None:
                self.scheduler.logger = self.logger
        else:
            self.logger = None

        if profiler:
            self.profiler = profiling.Profiler()
            if self.scheduler is not None:
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
        '''Make all the folder for the current branch according to :code:`self.paths`.
        '''
        for path in self.paths:
            mpi_mkdir(self.get_path(path))


    @property
    def paths(self):
        '''List of the name of all folders
        '''
        return [key for key in self.steps] + ['logs']


    @property
    def log_path(self):
        '''Path to the current log-output folder
        '''
        if self.persistancy:
            return self.root / self.branch_name / 'logs'
        else:
            return None



    def get_path(self, name=None):
        '''Given a relative file path, get the absolute path including root and branch.
        '''
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


    def branch(self, name, empty=False, linkfiles=None):
        '''Create branch by creating a copy of the current branch state and checkout that branch. If the branch exists, just checkout that branch.

        :param str name: Name of the new branch
        :param bool empty: If :code:`True` do not copy the state of the current branch.
        :param list/bool linkfiles: If a list of paths that should be soft-linked rather then copied. If :code:`True`, soft-link all files.
        :return: None
        '''
        if not self.persistancy:
            raise ValueError('Cannot use branches if no root is given')

        if (self.root / name).is_dir():
            if self.logger is not None:
                self.logger.info(f'Branch "{name}" already exists')
        else:
            if empty:
                mpi_mkdir(self.root / name)
                for path in self.paths:
                    mpi_mkdir(self.root / name / path)
            else:
                if linkfiles is None:
                    mpi_copy(self.root / self.branch_name, self.root / name, linkfiles=False)
                elif isinstance(linkfiles, list):
                    listing = pathlib.Path(self.root / self.branch_name).glob('./*')
                    for pth in listing:
                        if pth.name in linkfiles:
                            mpi_copy(pth, self.root / name / pth.name, linkfiles=True)
                        else:
                            mpi_copy(pth, self.root / name / pth.name, linkfiles=False)
                elif linkfiles:
                    mpi_copy(self.root / self.branch_name, self.root / name, linkfiles=True)
                else:
                    raise TypeError(f'linkfiles type "{type(linkfiles)}" not supported')

        # Make sure log directory exists
        mpi_mkdir(self.root / name / 'logs')

        self.checkout(name)


    def status(self):
        '''Prints the status
        '''
        raise NotImplementedError('should contain branch-list, branch sizes, step lists and logs avalible')


    def checkout(self, branch):
        '''Change to branch.
        '''
        self.branch_name = branch
        if self.logger is not None and self.persistancy:
            self.logger = profiling.change_logfile(self.logger, self.log_path)


    @log_exceptions
    def run(self, step = None, *args, **kwargs):
        '''Run specific step with the supplied arguments or run all steps'''

        if self.persistancy:
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


    def add_cmd_argument(self, *args, **kwargs):
        '''Takes same arguments and keyword arguments as `parser.add_argument` but does not add the argument to the parser until `parse_cmd` is called.

        All arguments added here ends up as keyword arguments to the steps, the argument name used is always .
        '''
        self.__user_args.append((args, kwargs))



    def parse_cmd(self):
        '''Parses the arguments from a terminal command-line execution of the simulation and executes appropriately
        '''
        arg_parser = argparse.ArgumentParser(description='Simulation command-line interface')

        subparsers = arg_parser.add_subparsers(dest='sim_action', help='Actions')

        run_parser = subparsers.add_parser('run', help='Run simulation')
        cmd_parser = subparsers.add_parser('cmd', help='Simulation command')
        cmd_subparsers = cmd_parser.add_subparsers(dest='sim_cmd', help='Commands')

        if self.persistancy:
            run_parser.add_argument(
                'branch', 
                type=str, 
                help='Name of the branch to operate on',
            )
            branch_parser = cmd_subparsers.add_parser('branch', help='Branch simulation')
            branch_parser.add_argument(
                'name', 
                type=str, 
                help='Name of new branch',
            )
            branch_parser.add_argument(
                '--source', 
                type=str, 
                default='',
                help='Name of branch to clone onto new branch',
            )
            branch_parser.add_argument(
                '--link-branch', 
                action='store_true',
                help='Soft-link all current files towards [source] branch',
            )
            delete_parser = cmd_subparsers.add_parser('delete', help='Delete branch')
            delete_parser.add_argument(
                'name', 
                type=str, 
                help='Branch name to delete',
            )

        run_parser.add_argument(
            'step',
            type=str,
            help='Name of the step to execute, if "all" executes all steps.',
        )

        dests = []
        for args, kwargs in self.__user_args:
            run_parser.add_argument(*args, **kwargs)
            action = run_parser._actions[-1]
            if hasattr(action, 'dest'):
                dests.append(action.dest)

        args = arg_parser.parse_args()

        if args.sim_action == 'cmd':
            if args.sim_cmd == 'branch':
                if len(args.source) > 0:
                    self.checkout(args.source)
                    _empty = False
                else:
                    _empty = True

                if args.link_branch:
                    linkfiles = True
                else:
                    linkfiles = None

                self.branch(name = args.name, empty = _empty, linkfiles = linkfiles)
            elif args.sim_cmd == 'delete':
                self.delete(args.name)

        elif args.sim_action == 'run':

            if args.step == 'all':
                step = None
            else:
                step = args.step

            if self.persistancy:
                self.checkout(args.branch)

            run_kwargs = dict()
            for dest in dests:
                run_kwargs[dest] = getattr(args, dest)

            self.run(step=step, **run_kwargs)
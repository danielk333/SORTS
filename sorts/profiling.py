'''Sets up a logging framework that can be imported and used anywhere.

'''

import logging
import datetime
import pathlib
import time

class Profiler:
    '''Performance profiler class.

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

    '''
    def __init__(self, distribution=False):
        self.distribution = distribution
        self.exec_times = dict()
        self.start_times = dict()

    def start(self, name):
        '''Records a start time for named call.
        '''
        self.start_times[name] = time.time()

    def stop(self, name):
        dt = time.time() - self.start_times[name]
        self.start_times[name] = None

        if self.distribution:
            if name in self.exec_times:
                self.exec_times[name].append(dt)
            else:
                self.exec_times[name] = [dt]
        else:
            if name in self.exec_times:
                self.exec_times[name]['spent'] += dt
                self.exec_times[name]['num'] += 1
            else:
                self.exec_times[name] = {'spent': dt, 'num': 1}

    def mean(self,name=None):
        if name is not None:
            if self.distribution:
                return np.mean(np.array(self.exec_times[name]))
            else:
                return self.exec_times[name]['spent']/self.exec_times[name]['num']
        else:
            ret = dict()
            for key in self.exec_times:
                if self.distribution:
                    ret[key] = np.mean(np.array(self.exec_times[key]))
                else:
                    ret[key] = self.exec_times[key]['spent']/self.exec_times[key]['num']
            return ret

    def total(self,name=None):
        if name is not None:
            if self.distribution:
                return np.sum(np.array(self.exec_times[name]))
            else:
                return self.exec_times[name]['spent']
        else:
            ret = dict()
            for key in self.exec_times:
                if self.distribution:
                    ret[key] = np.sum(np.array(self.exec_times[key]))
                else:
                    ret[key] = self.exec_times[key]['spent']
            return ret


    def __str__(self):
        max_key_len = 5
        for key in self.exec_times:
            if len(key) > max_key_len:
                max_key_len = len(key)
        header_ = f'{"Name":<{max_key_len}}| Executions | Mean time | Total time'



        str_ = ''
        str_ += f'{" Performance analysis ".center(len(header_), "-")}\n'
        str_ += header_ + '\n'
        str_ += '-'*len(header_) + '\n'

        means = self.mean()
        totals = self.total()

        for key in self.exec_times:
            if self.distribution:
                num = str(len(self.exec_times[key]))
            else:
                num = str(self.exec_times[key]['num'])

            if means[key] < 60:
                mu = f'{means[key]:.5e}'
            else:
                mu = str(datetime.timedelta(seconds=means[key]))
            if totals[key] < 60:
                su = f'{totals[key]:.5e}'
            else:
                su = str(datetime.timedelta(seconds=totals[key]))
            
            str_ += f'{key:<{max_key_len}}| {num:<10} | {mu:<9} | {su} \n'
        return str_



def add_logging_level(num, name):
    def fn(self, message, *args, **kwargs):
        if self.isEnabledFor(num):
            self._log(num, message, args, **kwargs)
    logging.addLevelName(num, name)
    setattr(logging, name, num)
    return fn


logging.Logger.always = add_logging_level(100, 'ALWAYS')


def get_logger(
        name = 'sorts',
        path = None,
        file_level = logging.INFO,
        term_level = logging.INFO,
    ):
    '''Returns a logger object
    
    Formats to output both to terminal and a log file
    '''
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_at_%H-%M")
    datefmt = '%Y-%m-%d %H:%M:%S'
    msecfmt = '%s.%03d'

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size > 1:
            parallel = comm.rank
        else:
            parallel = None
    except ImportError:
        parallel = None



    if path is not None:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if path.is_dir():
            if parallel is None:
                log_fname = f'{name}_{datetime_str}.log'
            else:
                log_fname = f'{name}_{datetime_str}_process{parallel}.log'
        else:
            if parallel is None:
                log_fname = str(path)
            else:
                new_name = path.name.replace(path.suffix, '') + f'_process{parallel}'
                log_fname = path.parent / f'{new_name}{path.suffix}'


    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if parallel is not None:
        format_str = f'%(asctime)s PID{parallel} %(levelname)-8s; %(message)s'
    else:
        format_str = '%(asctime)s %(levelname)-8s; %(message)s'

    if path is not None:
        fh = logging.FileHandler(log_fname)
        fh.setLevel(file_level) #debug and worse
        form_fh = logging.Formatter(format_str)
        form_fh.default_time_format = datefmt
        form_fh.default_msec_format = msecfmt
        fh.setFormatter(form_fh)
        logger.addHandler(fh) #id 0

    ch = logging.StreamHandler()
    ch.setLevel(term_level)
    form_ch = logging.Formatter(format_str)
    form_ch.default_time_format = datefmt
    form_ch.default_msec_format = msecfmt

    ch.setFormatter(form_ch)
    logger.addHandler(ch) #id 1

    return logger


def term_level(logger, level):
    if len(logger.handlers) == 1:
        term_id = 0
    else:
        term_id = 1

    if isinstance(level, str):
        level = getattr(logging, level)

    logger.handlers[term_id].setLevel(level)

    return logger


def file_level(logger, level):
    if len(logger.handlers) == 1:
        return logger
    else:
        file_id = 0

    if isinstance(level, str):
        level = getattr(logging, level)

    logger.handlers[file_id].setLevel(level)

    return logger
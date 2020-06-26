'''Sets up a logging framework that can be imported and used anywhere.

'''

import logging
import datetime
import time

exec_times = {'last': time.time()}

def record_time_diff(name):
    '''Records a time difference since last call
    
    This function modifies a global variable 'exec_times' in this module!
    This is especcialy useful for timing contents of loops

    example:
    .. code:python

        record_time_diff('loop_start')
        for i in range(large_number):
            function_one(*args)
            record_time_diff('function_one')

            function_two(*args)
            record_time_diff('function_two')
    '''
    global exec_times

    time_now = time.time()
    dt = time_now - exec_times['last']

    if name in exec_times:
        t_sum, num = exec_times[name]    
        exec_times[name] = (t_sum + dt, num + 1)
    else:
        exec_times[name] = (dt, 1)
    
    exec_times['last'] = time_now

def logg_time_record(exec_t, logger):
    '''Saves time record to log at info level
    '''
    logger.info('--------- TIME ANALYSIS --------')
    logger.info('PLACE   | PASSES | Mean time [s]')
    logger.info('--------------------------------')

    max_key_len = 0
    for key in exec_times:
        if len(key) > max_key_len:
            max_key_len = len(key)

    for key, val in exec_times.items():
        if key != 'last':
            t_sum, num = val
            logger.info('{:<{}}: n={:03d} @ {:.5e} s (total {:.2e} s)'\
                .format(key, max_key_len, num, t_sum/num, t_sum))


def extract_format_strings(form):
    '''Extracts the formatting string inside curly braces by returning the index positions
    
    For example:

    string = "{2|number_of_sheep} sheep {0|has} run away"
    form_v = extract_format_strings(string)
    print(form_v)
    for x,y in form_v:
        print(string[x:y])

    gives:
        2|number_of_sheep
        0|has
    '''

    form = ' '+form
    form_v = []
    ind = 0
    while ind < len(form) and ind >= 0:
        ind_open = form.find('{',ind+1)
        ind = ind_open
        if ind_open > 0:
            ind_close = form.find('}',ind+1)
            if ind_close > 0:
                form_v.append((ind_open, ind_close-1))
            ind = ind_close

    return form_v

def extract_format_keys(form):
    '''This function looks for our special formatting of indicating index of argument and name of argument
    
    Returns a list of tuples where each tuple is the key in index format and in named format
    '''
    form_inds = extract_format_strings(form)

    keys = []
    for start, stop in form_inds:
        form_arg = form[start:stop]

        if form_arg.index('|') > 0:
            keys.append(tuple(form_arg.split('|')))
        else:
            raise Exception('Log call formating only works with key | formating')

    return form_inds, keys

def construct_formatted_format(form, args_len, kwargs):
    '''This takes a special formatted string, extracts the two possible keys, 
        and based on what was passed as key-word arguments and what was passed as indexed arguemnts, 
        chooses the correct format. 
        If a option has a default value it will indicate this and not report a value.

        Returns a correctly formatted string for the input arguemts of the function.
    '''
    form_inds, keys = extract_format_keys(form)

    form_str = ''
    cntr = 0
    last_ind = 0

    for start,stop in form_inds:
        form_str += form[last_ind : start]

        ind, key = keys[cntr]

        if key.find('.') > 0:
            key_check = key[:key.find('.')]
        else:
            key_check = key

        if key_check in kwargs:
            form_str += key
            form_str += '}'
        else:
            if ind.find('.') > 0:
                ind_check = int(ind[:ind.find('.')])
            else:
                ind_check = int(ind)

            if ind_check > args_len-1:
                form_str = form_str[:-1]
                form_str += '[' + key + ':default]'
            else:
                form_str += ind
                form_str += '}'

        last_ind = stop+1
        cntr += 1
    form_str += form[last_ind :]

    return form_str


def log_call(form, logger):

    def log_call_decorator(method):
        def logged_fn(*args, **kwargs):

            form_str = construct_formatted_format(
                form,
                len(args),
                kwargs
                )

            logger.always('{}: {}'.format(
                method.__name__, 
                form_str.format(*args, **kwargs)
                ))
            return method(*args, **kwargs)

        return logged_fn
    return log_call_decorator

def class_log_call(form):

    def log_call_decorator(method):
        def logged_fn(*args, **kwargs):

            form_str = construct_formatted_format(
                form,
                len(args),
                kwargs
                )

            args[0].logger.always('{}.{}: {}'.format(
                repr(args[0]),
                method.__name__, 
                form_str.format(*args, **kwargs)
                ))
            return method(*args, **kwargs)

        return logged_fn
    return log_call_decorator




def add_logging_level(num, name):
    def fn(self, message, *args, **kwargs):
        if self.isEnabledFor(num):
            self._log(num, message, args, **kwargs)
    logging.addLevelName(num, name)
    setattr(logging, name, num)
    return fn

logging.Logger.always = add_logging_level(100, 'ALWAYS')

def setup_logging(
        name = 'SORTS++',
        root = '',
        file_level = logging.INFO,
        term_level = logging.INFO,
        parallel = 0,
        logfile = True,
    ):
    '''Returns a logger object to be used in simulations
    
    Formats to output both to terminal and a log file
    '''

    if len(root) > 0 and root[-1] != '/':
        root += '/'

    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_at_%H-%M")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logfile:
        log_fname = root\
            + 'SORTSpp_'\
            + datetime_str\
            + '_thread' + str(parallel)\
            + '.log'

        fh = logging.FileHandler(log_fname)
        fh.setLevel(file_level) #debug and worse
        form_fh = logging.Formatter(
            '%(asctime)s PID{} %(levelname)-8s; %(message)s'.format(parallel),
            '%Y-%m-%d %H:%M:%S'
            )
        fh.setFormatter(form_fh)
        logger.addHandler(fh) #id 0

    ch = logging.StreamHandler()
    ch.setLevel(term_level) #warnings and worse
    form_ch = logging.Formatter('PID{} %(levelname)-8s; %(message)s'.format(parallel))
    ch.setFormatter(form_ch)
    logger.addHandler(ch) #id 1


    return logger

if __name__=='__main__':
    '''
    CRITICAL    50
    ERROR   40
    WARNING 30
    INFO    20
    DEBUG   10
    '''
    #this tests logger
    record_time_diff('start')

    logger = setup_logging()

    record_time_diff('logger_setup')

    for i in range(10):
        # 'application' code
        logger.debug('debug message {}'.format(i))
        logger.info('info message {}'.format(i))
        logger.warn('warn message {}'.format(i))
        logger.error('error message {}'.format(i))
        logger.critical('critical message {}'.format(i))
        logger.always('always message {}'.format(i))
        record_time_diff('log loop')

    logg_time_record(exec_times,logger)

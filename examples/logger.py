#!/usr/bin/env python

'''
=======
Logging
=======

Showcases the use of teh ``sorts.logger`` module to display internal status/messages on 
the terminal.

The different message levels are :
     - CRITICAL     50
     - ERROR        40
     - WARNING      30
     - INFO         20
     - DEBUG        10
'''

import sorts

p = sorts.profiling.Profiler()

logger = sorts.profiling.get_logger('example')

for i in range(10):
    # 'application' code
    p.start('log')
    logger.debug('debug message {}'.format(i))
    logger.info('info message {}'.format(i))
    logger.warning('warning message {}'.format(i))
    logger.error('error message {}'.format(i))
    logger.critical('critical message {}'.format(i))
    logger.always('always message {}'.format(i))
    p.stop('log')

for line in str(p).split('\n'):
    logger.info(line)

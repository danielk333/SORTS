#!/usr/bin/env python

'''
Logging
==========


CRITICAL    50
ERROR   40
WARNING 30
INFO    20
DEBUG   10

'''

import logging

logger = logging.getLogger(__name__)

for i in range(10):
    # 'application' code
    logger.debug('debug message {}'.format(i))
    logger.info('info message {}'.format(i))
    logger.warning('warning message {}'.format(i))
    logger.error('error message {}'.format(i))
    logger.critical('critical message {}'.format(i))

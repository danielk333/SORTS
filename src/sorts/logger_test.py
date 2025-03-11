import logging
from .logger import config_logger


def test_config_logger():
    logger = logging.getLogger(__name__)
    config_logger(logger)

    print(f"this is a print message")
    logger.info(f"this is a info log message")

import logging
from .logging import apply_suggested_config


def test_apply_suggested_config():
    logger = logging.getLogger(__name__)
    apply_suggested_config(logger)

    print(f"this is a print message")
    logger.info(f"this is a info log message")

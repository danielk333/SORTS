import logging
from .logging import apply_suggested_config


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def test_apply_suggested_config():
    logger = logging.getLogger(__name__)
    apply_suggested_config(logger)

    print(f"this is a print message")
    logger.info(f"this is a info log message")

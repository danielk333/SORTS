import logging


# NOTE: here are some notes from danielk on logger setup/config
#   we should see if the way we configure logging by default is good or not and refactor that as
#   needed, e.g. if we look at
#   https://github.com/danielk333/runningman/blob/main/src/runningman/manager.py#L125
#   this is how i setup logging there, i think this code is similar
#   We should aim for the pattern of: if the user wants to log they can setup logging themself or
#   use a convenience function that does `setup_logging(bunch of useful parameters)` and it will configure
#   most things
def apply_suggested_config(logger: logging.Logger):
    """configure a logger to a suggested configuration"""

    handler = logging.StreamHandler()  # this goes to console
    formatter = logging.Formatter("[%(asctime)s]%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

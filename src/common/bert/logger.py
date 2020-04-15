import logging

INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING
ERROR = logging.ERROR


def get_logger(level=INFO):
    format = ("[%(levelname)s]: ""%(message)s")

    logging.basicConfig(level=level, format=format)
    logger = logging.getLogger(__name__)

    return logger

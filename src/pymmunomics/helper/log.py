"""Helper module for logging operations.

Constants
---------
LOG_HANDLER: logging.StreamHandler
    Handler used for logging.
LOGGER: logging.Logger
    Multiprocessing-safe logger.
"""
from logging import (
    captureWarnings,
    getLogger,
    Formatter,
    StreamHandler,
)
from multiprocessing import get_logger

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
LOGGING_FORMAT = (
    "%(asctime)s\t(%(processName)s, %(threadName)s)\t%(levelname)s\t%(message)s"
)

LOGGER = get_logger()
formatter = Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT)
LOG_HANDLER = StreamHandler()
LOG_HANDLER.setFormatter(formatter)
LOGGER.addHandler(LOG_HANDLER)

captureWarnings(True)
getLogger("py.warnings").addHandler(LOG_HANDLER)

def set_log_level(level):
    LOGGER.setLevel(level)
    # if LOGGER.level > 30:
    #     captureWarnings(False)

import logging
import sys

from app.core.config import settings


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("palmmind")

    if logger.handlers:
        return logger

    log_level = logging.DEBUG if settings.env == "development" else logging.INFO
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()

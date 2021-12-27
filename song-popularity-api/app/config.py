import logging
import sys
from types import FrameType
from typing import cast

from loguru import logger
from pydantic import BaseSettings


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    PROJECT_NAME: str = "Song Popularity Prediction API"

    HOST: str = "localhost"
    PORT: int = 8001

    logging: LoggingSettings = LoggingSettings()


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:

        # ignore messages from PropagateHandler to avoid hanging
        if getattr(record, "from_loguru", False):
            return

        # get corresponding loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_app_logging(config: Settings) -> None:
    """Prepare custom logging for our application"""

    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    )


settings = Settings()

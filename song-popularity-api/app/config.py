import logging
from pydantic import BaseSettings


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    PROJECT_NAME: str = "Song Popularity Prediction API"

    HOST: str = "localhost"
    PORT: int = 8001

    logging: LoggingSettings = LoggingSettings()


settings = Settings()

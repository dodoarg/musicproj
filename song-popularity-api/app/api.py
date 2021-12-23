from fastapi import APIRouter

from classification_model import __version__ as model_version

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Root Get"""

    health = schemas.Health(
        name=settings.PROJECT_NAME,
        api_version = __version__,
        model_version = model_version
    )

    return health.dict()
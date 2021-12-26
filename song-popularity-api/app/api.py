import json
from typing import Any
from loguru import logger

import numpy as np
import pandas as pd
from classification_model import __version__ as model_version
from classification_model.predict import make_prediction
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Root Get"""

    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleSongsDataInputs) -> Any:
    """Make popularity predictions with the dodoarg-hit-song-science model"""

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    logger.info(f"Making predictions on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    # I can't think of a case where error 422 is not raised as a result of input_data
    # not matching the schema before make_predictions is even called
    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))
    if isinstance(results["predictions"], np.ndarray):
        results["predictions"] = results["predictions"].tolist()

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results

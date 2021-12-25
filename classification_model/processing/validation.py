from typing import List, Optional, Sequence, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop features that have missing values but are not
    accounted for in the pipeline
    """
    validated_data = input_data.copy()
    vars_to_drop = [
        var for var in validated_data.columns if validated_data[var].isnull().sum() > 0
    ]

    validated_data.dropna(subset=vars_to_drop, inplace=True)

    return validated_data


def validate_inputs(
    *,
    input_data: pd.DataFrame,
    _vars_to_cast_as_obj: Sequence[str] = config.model_config.categorical_features,
    _relevant_features: Sequence[str] = config.model_config.features
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values"""

    input_data[_vars_to_cast_as_obj] = input_data[_vars_to_cast_as_obj].astype("O")

    relevant_data = input_data[_relevant_features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        MultipleSongsDataInputs(inputs=validated_data.to_dict(orient="records"))

    except ValidationError as error:
        errors = error.errors()

    return validated_data, errors


class SongsDataInputSchema(BaseModel):
    acousticness: Optional[float]
    danceability: Optional[float]
    energy: Optional[float]
    instrumentalness: Optional[float]
    key: Optional[int]
    liveness: Optional[float]
    loudness: Optional[float]
    mode: Optional[int]
    speechiness: Optional[float]
    tempo: Optional[float]
    time_signature: Optional[int]
    valence: Optional[float]
    beats_count: Optional[int]
    chroma_stft_mean: Optional[float]
    root_mean_square_mean: Optional[float]
    spectral_centroid_mean: Optional[float]
    spectral_bandwidth_mean: Optional[float]
    rolloff_mean: Optional[float]
    zero_crossing_rate_mean: Optional[float]
    mfcc_1_mean: Optional[float]
    mfcc_2_mean: Optional[float]
    mfcc_3_mean: Optional[float]
    mfcc_4_mean: Optional[float]
    mfcc_5_mean: Optional[float]
    mfcc_6_mean: Optional[float]
    mfcc_7_mean: Optional[float]
    mfcc_8_mean: Optional[float]
    mfcc_9_mean: Optional[float]
    mfcc_10_mean: Optional[float]
    mfcc_11_mean: Optional[float]
    mfcc_12_mean: Optional[float]
    mfcc_13_mean: Optional[float]
    mfcc_14_mean: Optional[float]
    mfcc_15_mean: Optional[float]
    mfcc_16_mean: Optional[float]
    mfcc_17_mean: Optional[float]
    mfcc_18_mean: Optional[float]
    mfcc_19_mean: Optional[float]
    mfcc_20_mean: Optional[float]


class MultipleSongsDataInputs(BaseModel):
    inputs: List[SongsDataInputSchema]

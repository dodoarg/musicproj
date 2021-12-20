from typing import List, Optional, Sequence, Tuple

import pandas as pd
from pydantic import BaseModel, StrictFloat, StrictStr, ValidationError

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
    acousticness: Optional[StrictFloat]
    danceability: Optional[StrictFloat]
    energy: Optional[StrictFloat]
    instrumentalness: Optional[StrictFloat]
    key: Optional[StrictStr]
    liveness: Optional[StrictFloat]
    loudness: Optional[StrictFloat]
    mode: Optional[StrictStr]
    speechineess: Optional[StrictFloat]
    tempo: Optional[StrictFloat]
    time_signature: Optional[StrictStr]
    valence: Optional[StrictFloat]
    beats_count: Optional[StrictFloat]
    chroma_stft_mean: Optional[StrictFloat]
    root_mean_square: Optional[StrictFloat]
    spectral_centroid_mean: Optional[StrictFloat]
    spectral_bandwidth_mean: Optional[StrictFloat]
    rolloff_mean: Optional[StrictFloat]
    zero_crossing_rate_mean: Optional[StrictFloat]
    mfcc_1_mean: Optional[StrictFloat]
    mfcc_2_mean: Optional[StrictFloat]
    mfcc_3: Optional[StrictFloat]
    mfcc_4_mean: Optional[StrictFloat]
    mfcc_5_mean: Optional[StrictFloat]
    mfcc_6_mean: Optional[StrictFloat]
    mfcc_7_mean: Optional[StrictFloat]
    mfcc_8_mean: Optional[StrictFloat]
    mfcc_9_mean: Optional[StrictFloat]
    mfcc_10_mean: Optional[StrictFloat]
    mfcc_11_mean: Optional[StrictFloat]
    mfcc_12_mean: Optional[StrictFloat]
    mfcc_13_mean: Optional[StrictFloat]
    mfcc_14_mean: Optional[StrictFloat]
    mfcc_15_mean: Optional[StrictFloat]
    mfcc_16_mean: Optional[StrictFloat]
    mfcc_17_mean: Optional[StrictFloat]
    mfcc_18_mean: Optional[StrictFloat]
    mfcc_19_mean: Optional[StrictFloat]
    mfcc_20_mean: Optional[StrictFloat]


class MultipleSongsDataInputs(BaseModel):
    inputs: List[SongsDataInputSchema]

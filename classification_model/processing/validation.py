from typing import Optional, Sequence, Tuple

import pandas as pd

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

    return validated_data, errors

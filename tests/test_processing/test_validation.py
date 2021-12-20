import numpy as np
import pandas as pd

from classification_model.processing.validation import validate_inputs


def test_validate_inputs():
    df = pd.DataFrame(
        {
            "var1": [0, 1, 2, 3, 4],
            "var2": [4, 3, 2, 1, 0],
            "var3": [1.24, np.NaN, 9.98, 0.12, -12.11],
        }
    )

    exp_validated_df = df.copy()
    exp_validated_df["var2"] = exp_validated_df["var2"].astype("O")
    exp_validated_df.drop("var1", axis=1, inplace=True)
    exp_validated_df.drop(1, axis=0, inplace=True)

    validated_df, errors = validate_inputs(
        input_data=df,
        _vars_to_cast_as_obj=["var2"],
        _relevant_features=["var2", "var3"],
    )
    pd.testing.assert_frame_equal(validated_df, exp_validated_df)
    assert errors is None

    # catch pydantic validation errors
    df_not_valid = pd.DataFrame(
        {
            "instrumentalness": [0.24, 0.11, -2.34],
            "energy": ["not_a_float", 1.24, 93.23],
            "key": ["key_0", 1, "key_2"],
        }
    )
    exp_error_energy = {
        "loc": ("inputs", 0, "energy"),
        "msg": "value is not a valid float",
        "type": "type_error.float",
    }
    exp_error_key = {
        "loc": ("inputs", 1, "key"),
        "msg": "str type expected",
        "type": "type_error.str",
    }
    exp_error = [exp_error_energy, exp_error_key]
    validated_df, errors = validate_inputs(
        input_data=df_not_valid,
        _vars_to_cast_as_obj=[],
        _relevant_features=["instrumentalness", "energy", "key"],
    )
    pd.testing.assert_frame_equal(validated_df, df_not_valid)
    assert errors == exp_error

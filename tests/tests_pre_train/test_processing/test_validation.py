import numpy as np
import pandas as pd

from classification_model.processing.validation import validate_inputs


def test_validate_toy_inputs_data():
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
            "key": [10, 3, "not_an_int"],
        }
    )
    exp_error_energy = {
        "loc": ("inputs", 0, "energy"),
        "msg": "value is not a valid float",
        "type": "type_error.float",
    }
    exp_error_key = {
        "loc": ("inputs", 2, "key"),
        "msg": "value is not a valid integer",
        "type": "type_error.integer",
    }
    exp_error = [exp_error_energy, exp_error_key]
    validated_df, errors = validate_inputs(
        input_data=df_not_valid,
        _vars_to_cast_as_obj=[],
        _relevant_features=["instrumentalness", "energy", "key"],
    )
    pd.testing.assert_frame_equal(validated_df, df_not_valid)
    assert errors == exp_error


def test_validate_real_input_data(sample_input_data):
    input_data = sample_input_data.copy()

    validated_data, errors = validate_inputs(input_data=input_data)

    # one row contains missing data
    assert len(sample_input_data) == 500
    assert len(validated_data) == 499
    assert errors is None

    # introduce errors
    input_data["root_mean_square_mean"] = input_data["root_mean_square_mean"].astype(
        "O"
    )
    input_data.at[47, "root_mean_square_mean"] = "not_a_number"
    validated_data, errors = validate_inputs(input_data=input_data)
    exp_errors = [
        {
            "loc": ("inputs", 46, "root_mean_square_mean"),
            "msg": "value is not a valid float",
            "type": "type_error.float",
        }
    ]

    # we expect rms_mean to be floats
    assert errors is not None
    assert errors == exp_errors

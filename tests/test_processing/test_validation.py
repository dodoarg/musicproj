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

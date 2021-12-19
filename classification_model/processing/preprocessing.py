import numpy as np
import pandas as pd


def binarize_popularity(popularity: pd.Series) -> pd.Series:
    _, bins = pd.qcut(popularity, 4, retbins=True)
    seventyfive_percentile = bins[3]
    is_popular = popularity.apply(lambda x: np.where(x >= seventyfive_percentile, 1, 0))
    return is_popular


def balance_dataset(data: pd.DataFrame, target_bin: str) -> pd.DataFrame:
    n_popular = data[target_bin].sum()
    ind_to_drop = np.random.choice(
        data[data[target_bin] == 0].index, data.shape[0] - 2 * n_popular, replace=False
    )
    return data.drop(ind_to_drop)

import pandas as pd
import numpy as np

def binarize_popularity(popularity: pd.Series) -> pd.Series:
    _, bins = pd.qcut(popularity, 4, retbins=True)
    seventyfive_percentile = bins[3]
    is_popular = popularity.apply(
        lambda x: np.where(x >= seventyfive_percentile, 1, 0)
    )
    return is_popular

def balance_dataset(data: pd.DataFrame) -> pd.DataFrame:
    n_popular = data.is_popular.sum()
    ind_to_drop = np.random.choice(
        data[data.is_popular==0].index,
        data.shape[0] - 2*n_popular,
        replace=False
    )
    return data.drop(ind_to_drop)

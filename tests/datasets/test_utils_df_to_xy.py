import numpy as np
import pandas as pd

from merlin.datasets.utils import df_to_xy


def test_df_to_xy_default_last_column_is_label():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5], "label": [0, 1, 0]})

    X, y = df_to_xy(df)

    assert X.shape == (3, 2)
    assert y.shape == (3,)
    assert np.array_equal(X, df[["a", "b"]].to_numpy())
    assert np.array_equal(y, df[["label"]].to_numpy().ravel())


def test_df_to_xy_with_explicit_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]})

    X, y = df_to_xy(df, feature_cols=["a"], label_cols=["b", "label"])

    assert X.shape == (2, 1)
    assert y.shape == (2, 2)
    assert np.array_equal(X, df[["a"]].to_numpy())
    assert np.array_equal(y, df[["b", "label"]].to_numpy())

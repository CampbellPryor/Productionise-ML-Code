import pandas as pd
import numpy as np
from src.helpers.helpers import train_test_split


def test_train_test_split():
    df = pd.DataFrame(
        {
            "col1": np.arange(12),
            "col2": np.arange(12),
        }
    )
    train_data, validation_data, test_data = train_test_split(
        df, train_frac=0.5, validation_frac=0.33
    )
    assert set(df.columns) == set(train_data.columns)
    assert len(train_data) == 6
    assert len(validation_data) == 4
    assert len(test_data) == 2

    train_data, validation_data, test_data = train_test_split(
        df, train_frac=0.5, validation_frac=0.5
    )
    assert len(train_data) == 6
    assert len(validation_data) == 6
    assert len(test_data) == 0

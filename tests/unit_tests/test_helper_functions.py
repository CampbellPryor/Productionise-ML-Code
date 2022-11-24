import os
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sagemaker.inputs import TrainingInput
from unittest import mock
from unittest.mock import patch


from src.helpers.helpers import (
    load_params,
    load_data,
    drop_columns,
    series_to_object,
    get_dummies,
    save_data_for_sagemaker,
    point_to_data_location,
    get_default_bucket,
    get_user_role,
    download_file,
    get_training_container,
)


def test_load_params(env="DEV"):
    file_location = "config/config.yaml"
    params = load_params(file_location)

    assert type(params) == dict
    assert "SOURCE" in params
    assert "OUTPUT" in params
    assert "SETTINGS" in params
    assert "HYPERPARAMETERS" in params
    assert len(params["OUTPUT"]["output_folder"]) > 0


def test_load_params_prod():
    test_load_params(env="PROD")


def test_load_data():
    df = load_data("tests/assets/example_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert is_string_dtype(df["textheading"])
    assert is_numeric_dtype(df["numericheading"])


def test_drop_columns():
    test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    df = drop_columns(test_df, "col2")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(["col1"])
    assert len(df) == len(test_df)


def test_series_to_object():
    s_test = pd.Series([1502, 2101, 3310])
    s = series_to_object(s_test)
    assert s.dtype == object
    assert s[0] == "1502"


def test_get_dummies():
    df_test = pd.DataFrame(
        {"numeric_col": [1, 2, 3, 4], "obj_col": ["maybe", "yes", "no", "yes"]}
    )

    df = get_dummies(df_test, "obj_col")
    print(df)
    assert isinstance(df, pd.DataFrame)
    assert is_numeric_dtype(df["numeric_col"])
    assert df_test["obj_col"].nunique() - 1 == len(
        [x for x in df.columns if x != "numeric_col"]
    )
    assert df["obj_col_maybe"].max() == 1


def test_save_data_for_sagemaker(tmp_path):
    """In this test, we use pytest's tmp_path fixture to
    allow pytest to create a temporary directory for us
    to save files to.
    """
    df = pd.DataFrame(
        {
            "col1": np.arange(12),
            "col2": np.arange(12),
        }
    )
    path_to_file = tmp_path / "temp_file.csv"
    save_data_for_sagemaker(df, path_to_file)

    assert os.path.exists(path_to_file)


def test_point_to_data_location():

    s3_input = point_to_data_location("an_output_path", "sub_folder_name")
    print(s3_input)
    print(type(s3_input))

    assert isinstance(s3_input, TrainingInput)


"""@mock.patch("src.helpers.helpers.sagemaker")
def test_get_training_container(mock_sagemaker):
    mock_sagemaker.boto_region_name.return_value = "us-east-1"
    with patch("src.helpers.helpers.sagemaker.boto_region_name", "us-east-1"):
        container = get_training_container()
    mock_sagemaker.image_uris.retrieve.assert_called_with(
        "xgboost", "us-east-1", "1.5-1"
    )"""

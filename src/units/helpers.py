import re
import os
import json
import boto3
import pandas as pd
import numpy as np
import sagemaker


def load_params(config_location):

    # Load static params
    with open(config_location) as json_file:
        params = json.load(json_file)
    params["output_folder"] = f"s3://{params['bucket']}/{params['prefix']}"
    return params


def get_default_bucket():
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    return bucket


def get_user_role():
    role = sagemaker.get_execution_role()
    return role


def download_file(source_location, target_location, source_bucket=None):
    s3 = boto3.client("s3")
    s3.download_file(source_bucket, source_location, target_location)
    return None


def load_data(location):
    if location.endswith("csv"):
        df = pd.read_csv(location)
        return df
    raise ValueError("Only CSVs are currently supported")


def upload_file(bucket, source_loc, destination_loc):
    boto3.Session().resource("s3").Bucket(bucket).Object(destination_loc).upload_file(
        source_loc
    )


def drop_columns(df, column_names):
    df = df.drop(column_names, axis=1)
    return df


def series_to_object(series):
    series = series.astype(object)
    return series


def get_dummies(df):
    model_data = pd.get_dummies(df)
    # TODO Generalise this bit
    model_data = pd.concat(
        [
            model_data["Churn?_True."],
            model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
        ],
        axis=1,
    )
    return model_data


def train_test_split(model_data, train_frac, validation_frac):
    train_count = int(train_frac * len(model_data))
    validation_count = int(validation_frac * len(model_data))
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [train_frac, validation_count],
    )
    return train_data, validation_data, test_data


def save_data_for_sagemaker(df, filename):
    df.to_csv(filename, header=False, index=False)
    return None


def get_training_container():
    sess = sagemaker.Session()
    # TODO Define some of these settings in config file instead
    container = sagemaker.image_uris.retrieve("xgboost", sess.boto_region_name, "1.5-1")
    return container


def point_to_data_location(settings, folder_name):
    s3_input = TrainingInput(
        s3_data=f"{settings['output_folder']}/{folder_name}/", content_type="csv"
    )
    return s3_input

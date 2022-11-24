import re
import os
import json
import boto3
import yaml
import pandas as pd
import numpy as np
import sagemaker
from sagemaker.inputs import TrainingInput


def load_params(config_location, env="DEV"):

    # Load static params
    with open(config_location, "r") as stream:
        params = yaml.safe_load(stream)
    # Select out the DEV or PROD key based on env
    new_params = {key: val for key, val in params.items() if key not in ["DEV", "PROD"]}
    new_params.update(params[env])
    new_params["OUTPUT"][
        "output_folder"
    ] = f"s3://{new_params['OUTPUT']['bucket']}/{new_params['OUTPUT']['prefix']}"
    return new_params


def get_default_bucket():
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    return bucket


def get_user_role():
    try:
        iam = boto3.client("iam")
        role = iam.get_role(RoleName=os.environ.get("sagemaker_arn"))["Role"]["Arn"]
    except:
        role = sagemaker.get_execution_role()

    return role


def download_file(source_bucket, source_location, target_location):
    s3 = boto3.client("s3")
    s3.download_file(source_bucket, source_location, target_location)
    return None


def load_data(location):
    if location.endswith("csv"):
        df = pd.read_csv(location, parse_dates=True)
        return df
    raise ValueError("Only CSVs are currently supported")


def upload_file(bucket, source_loc, destination_loc):
    boto3.Session().resource("s3").Bucket(bucket).Object(destination_loc).upload_file(
        source_loc
    )


def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df


def series_to_object(series):
    series = series.apply(str)
    return series


def get_dummies(df, target_col):
    dummied_data = pd.get_dummies(df[target_col])
    dummied_data = dummied_data.iloc[:, :-1]

    dummied_data.columns = target_col + "_" + dummied_data.columns
    df = df.drop(columns=[target_col])
    df = df.join(dummied_data)

    return df


def train_test_split(model_data, train_frac, validation_frac):
    train_count = int(np.round(train_frac * len(model_data), 0))
    validation_count = int(np.round(validation_frac * len(model_data), 0))
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [train_count, train_count + validation_count],
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


def point_to_data_location(output_folder, folder_name="train"):
    s3_input = TrainingInput(
        s3_data=f"{output_folder}/{folder_name}/", content_type="csv"
    )
    return s3_input

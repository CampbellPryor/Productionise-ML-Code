import os
import sagemaker
import pandas as pd
import numpy as np
from sagemaker.serializers import CSVSerializer
from src.helpers import helpers


def load_settings(param_path: str, env: str = "DEV") -> dict:
    """Load settings from a config file and add some extra settings in

    Args:
        param_path: Path to a file containing parameters
        env: One of "DEV" or "PROD", representing whether to load settings
            for the development or production environment

    Returns:
        A dictionary containing the relevant settings
    """
    settings = helpers.load_params(param_path, env=env)
    settings["bucket"] = helpers.get_default_bucket()
    settings["role"] = helpers.get_user_role()
    return settings


def load_data(source_location, source_bucket):
    target_location = "temp.csv"
    print(f"BUCKET: {source_bucket}")
    helpers.download_file(
        source_bucket=source_bucket,
        source_location=source_location,
        target_location=target_location,
    )
    df = helpers.load_data(target_location)
    os.remove(target_location)
    return df


def clean_data(
    source_location,
    prefix,
    destination_bucket,
    source_bucket,
    train_frac,
    validation_frac,
):
    df = load_data(source_location, source_bucket)
    df = helpers.drop_columns(
        df,
        columns=[
            "Phone",
            "Day Charge",
            "Eve Charge",
            "Night Charge",
            "Intl Charge",
        ],
    )

    df["Area Code"] = helpers.series_to_object(df["Area Code"])

    model_data = helpers.get_dummies(df, "Churn?")

    train_data, validation_data, test_data = helpers.train_test_split(
        model_data, train_frac=train_frac, validation_frac=validation_frac
    )

    # Upload data to S3 to be used by sagemaker
    helpers.save_data_for_sagemaker(train_data, "train.csv")
    helpers.save_data_for_sagemaker(validation_data, "validation.csv")
    helpers.upload_file(
        bucket=destination_bucket,
        source_loc="train.csv",
        destination_loc=os.path.join(prefix, "train/train.csv"),
    )
    helpers.upload_file(
        bucket=destination_bucket,
        source_loc="validation.csv",
        destination_loc=os.path.join(prefix, "validation/validation.csv"),
    )
    return None


def train_model(settings):

    # Define training container
    container = helpers.get_training_container()

    # Point to data input location
    s3_input_train = helpers.point_to_data_location(
        output_folder=settings["output_folder"], folder_name="train"
    )
    s3_input_validation = helpers.point_to_data_location(
        output_folder=settings["output_folder"], folder_name="validation"
    )

    # Define estimator
    # TODO Could move these into their own helper functions
    xgb = sagemaker.estimator.Estimator(
        container,
        settings["role"],
        instance_count=settings["SETTINGS"]["instance_count"],
        instance_type=settings["SETTINGS"]["instance_type"],
        output_path=f"{settings['output_folder']}/output",
        sagemaker_session=sagemaker.Session(),
    )
    xgb.set_hyperparameters(**settings["HYPERPARAMETERS"])

    xgb.fit({"train": s3_input_train, "validation": s3_input_validation})
    return xgb


def host_model(xgb):
    xgb_predictor = xgb.deploy(
        initial_instance_count=1,
        instance_type="ml.m4.xlarge",
        serializer=CSVSerializer(),
    )
    return xgb_predictor


def get_predictions(xgb, data, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ""
    for array in split_array:
        predictions = "".join([predictions, xgb.predict(array).decode("utf-8")])

    return predictions.split("\n")[:-1]


def evaluate_performance(test_data, predictions):
    accuracy = (test_data == predictions).mean()
    return accuracy


def find_optimal_cutoff():
    # TODO Move this off into helper function
    cutoffs = np.arange(0.01, 1, 0.01)
    costs = []
    for c in cutoffs:
        costs.append(
            np.sum(
                np.sum(
                    np.array([[0, 100], [500, 100]])
                    * pd.crosstab(
                        index=test_data.iloc[:, 0],
                        columns=np.where(predictions > c, 1, 0),
                    )
                )
            )
        )

    costs = np.array(costs)
    return cutoffs[np.argmin(costs)]


def clean_up(xgb):
    xgb.delete_endpoint()

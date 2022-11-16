import os
from units import helpers
import pandas as pd


def load_settings(param_file):
    settings = helpers.load_params("config/config.yaml")
    settings["bucket"] = helpers.get_default_bucket()
    settings["role"] = helpers.get_user_role()
    return settings


def load_data(source_location, source_bucket=None):
    target_location = "temp.csv"
    helpers.download_file(source_location, source_bucket)
    df = helpers.load_data(target_location)
    os.remove(target_location)
    return df


def clean_data(source_location, prefix, desination_bucket, source_bucket=None):
    df = load_data(source_location, source_bucket)
    df = helpers.drop_columns(
        ["Phone", "Day Charge", "Eve Charge", "Night Charge", "Intl Charge"]
    )

    df["Area Code"] = helpers.series_to_object(df["Area Code"])

    model_data = helpers.get_dummies(df)

    train_data, validation_data, test_data = helpers.train_test_split(model_data)

    # Upload data to S3 to be used by sagemaker
    helpers.save_data_for_sagemaker(train_data, "train.csv")
    helpers.save_data_for_sagemaker(validation_data, "validation.csv")
    helpers.upload_file(
        bucket=desination_bucket,
        source_loc="train.csv",
        destination_loc=os.path.join(prefix, "train/train.csv"),
    )
    helpers.upload_file(
        bucket=desination_bucket,
        source_loc="validation.csv",
        destination_loc=os.path.join(prefix, "validation/validation.csv"),
    )
    return None


def train_model(settings):

    # Define training container
    container = helpers.get_training_container()

    # Point to data input location
    s3_input_train = helpers.point_to_data_location(settings=settings, folder="train")
    s3_input_validation = helpers.point_to_data_location(
        settings=settings, folder="validation"
    )

    # Define estimator
    xgb = sagemaker.estimator.Estimator(
        container,
        settings["role"],
        instance_count=1,
        instance_type="ml.m4.xlarge",
        output_path=f"{settings['output_folder']}/output",
        sagemaker_session=sess,
    )
    xgb.set_hyperparameters(
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.8,
        verbosity=0,
        objective="binary:logistic",
        num_round=100,
    )

    # fit
    xgb.fit({"train": s3_input_train, "validation": s3_input_validation})
    return xgb


def host_model(xgb):
    xgb_predictor = xgb.deploy(
        initial_instance_count=1,
        instance_type="ml.m4.xlarge",
        serializer=CSVSerializer(),
    )


def get_predictions(xgb, data, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ""
    for array in split_array:
        predictions = "".join([predictions, xgb.predict(array).decode("utf-8")])

    return predictions.split("\n")[:-1]


def evaluate_performance(test_data, predictions):
    crsstb = pd.crosstab(
        index=test_data.iloc[:, 0],
        columns=np.round(predictions),
        rownames=["actual"],
        colnames=["predictions"],
    )
    return crsstb


def find_optimal_cutoff():
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

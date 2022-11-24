import argparse
from src.steps import steps


def train_and_evaluate(config_path, env="DEV", cleanup=True):
    settings = steps.load_settings(config_path, env=env)

    (
        train_data_location,
        validation_data_location,
        test_data_location,
    ) = steps.clean_data(
        source_location=settings["SOURCE"]["location"],
        prefix=settings["OUTPUT"]["prefix"],
        destination_bucket=settings["OUTPUT"]["bucket"],
        source_bucket=settings["SOURCE"]["bucket"],
        train_frac=settings["SETTINGS"]["train_frac"],
        validation_frac=settings["SETTINGS"]["validation_frac"],
    )

    xgb = steps.train_model(train_data_location, validation_data_location)

    model_endpoint = steps.host_model(xgb)

    predictions = steps.predict(test_data_location)

    model_performance = steps.evaluate_model(test_data_location, predictions)

    return model_endpoint, model_performance


# Allow file to be called from command line with arguments passed in
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pipeline Arguments")
    parser.add_argument(
        "-c",
        "--config_path",
        dest="config_path",
        help="Path to Config File",
        required=True,
        default="config/config.yaml",
    )
    parser.add_argument(
        "-e",
        "--env",
        dest="env",
        help="Environment - one of DEV or PROD",
        required=False,
        default="DEV",
    )
    args = parser.parse_args()

    # Invoke the pipeline
    train_and_evaluate(
        config_path=args.config_path,
    )

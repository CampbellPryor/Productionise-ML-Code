from src.pipelines.train_and_evaluate import train_and_evaluate


def test_train_and_evaluate():
    model_endpoint, model_performance = train_and_evaluate(
        config_path="tests/assets/config.yaml"
    )
    assert model_performance <= 1 and model_performance >= 0

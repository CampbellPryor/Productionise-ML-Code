from steps import steps

settings = steps.load_settings("config/config.yaml")

train_data_location, validation_data_location, test_data_location = steps.clean_data()

xgb = steps.train_model(train_data_location, validation_data_location)

model_endpoint = steps.host_model(xgb)

predictions = steps.predict(test_data_location)

model_performance = steps.evaluate_model(test_data_location, predictions)

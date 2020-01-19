class KerasHost:
  """
  Host class for training/evaluation/inference of Keras based models
  This host sets up the EarlyStopping and Tensorboard callbacks
  """
  def train_model(training_env, data_env, build_model_fn):
    """build a model using the build_model_fn,
       compile that object
       fit the parameters of that object to data
       save the object to disk
    """
    from keras.callbacks import EarlyStopping, TensorBoard

    print("building model...")
    model = build_model_fn(training_env, data_env)

    print("compiling model...")
    model.compile(loss=training_env["loss"],
                  optimizer=training_env["optimizer"],
                  metrics=training_env["metrics"])

    print("fitting model...")
    model.fit(data_env["x-train"], data_env["y-train"],
              batch_size=training_env["batch-size"],
              epochs=training_env["num-epochs"],
              validation_data=(data_env["x-validation"], data_env["y-validation"]),
              verbose=2,
              callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5),
                         TensorBoard(log_dir=data_env["log-dir"])])

    print("saving model to '%s'" % data_env["model-filename"])
    model.save(data_env["model-filename"])
    print("saved model")

    return model

  def evaluate_model(data_env):
    """evaluate the saved model against the test set
    """
    from keras.models import load_model

    model = load_model(data_env["model-filename"])
    score, acc = model.evaluate(data_env["x-test"], data_env["y-test"],
                                batch_size=1)
    return score, acc

  def infer_with_model(model_filename, input_string, max_length, max_features):
    """evaluate the saved model against the test set
    """
    from keras.model import load_model

    model = load_model(model_filename)
    pred_string = project_text_to_imdb_data(input_string, max_features, max_length)
    print("predicting on '%s'" % str(input_string))
    print("predicting on '%s'" % str(pred_string))
    pred = model.predict(pred_string, batch_size=1)
    print("got: '%s'" % str(pred))
    return pred

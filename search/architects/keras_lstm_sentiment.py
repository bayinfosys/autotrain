from __future__ import print_function

from environments import (get_default_training_environment,
                          get_default_data_environment,
                          load_default_imdb_data)

def train_model(training_env, data_env):
  def build_model(model_env):
    from keras.models import Sequential
    from keras.layers import Dense, Embedding
    from keras.layers import LSTM

    #assert "max-features" in model_env
    #assert "sequence-length" in model_env

    model = Sequential()
    model.add(Embedding(data_env["max-features"], data_env["sequence-length"]))
    model.add(LSTM(data_env["sequence-length"], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model

  from keras.callbacks import EarlyStopping

  # build the model
  model = build_model(training_env)

  print(model.summary())

  # try using different optimizers and different optimizer configs
  model.compile(loss=training_env["loss"],
                optimizer=training_env["optimizer"],
                metrics=training_env["metrics"])

  model.fit(data_env["x-train"],
            data_env["y-train"],
            batch_size=training_env["batch-size"],
            epochs=training_env["num-epochs"],
            validation_data=(data_env["x-validation"], data_env["y-validation"]),
            callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)])

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

def infer_with_model(data_env):
  assert False

if __name__ == "__main__":
  from keras.preprocessing import sequence
  from keras.datasets import imdb
  from keras.models import load_model

  training_env = get_default_training_environment()
  data_env = get_default_data_environment()

  load_default_imdb_data(data_env,
                         max_features=training_env["max-features"],
                         max_length=training_env["max-length"])

  # build the model
  train_model(training_env, data_env)

  # evaluate the model
  model = load_model(training_env["model-filename"])
  score, acc = model.evaluate(data_env["x-test"], data_env["y-test"],
                              batch_size=training_env["batch-size"])
  print('Test score:', score)
  print('Test accuracy:', acc)


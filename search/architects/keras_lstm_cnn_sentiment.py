from __future__ import print_function

from environments import (get_default_training_environment,
                          get_default_data_environment,
                          load_default_imdb_data,
                          project_text_to_imdb_data)

def train_model(training_env, data_env):
  def build_model(model_env):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers import Embedding
    from keras.layers import LSTM
    from keras.layers import Conv1D, MaxPooling1D

    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size = 70

    model = Sequential()
    model.add(Embedding(data_env["max-features"],
                        data_env["sequence-length"],
                        input_length=data_env["max-length"]))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

  import json
  from keras.callbacks import EarlyStopping, TensorBoard

  print(json.dumps(training_env, indent=2))

  model = build_model(training_env)

  model.compile(loss=training_env["loss"],
                optimizer=training_env["optimizer"],
                metrics=training_env["metrics"])

  model.fit(data_env["x-train"], data_env["y-train"],
            batch_size=training_env["batch-size"],
            epochs=training_env["num-epochs"],
            validation_data=(data_env["x-validation"], data_env["y-validation"]),
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

if __name__ == "__main__":
  import sys
  from keras.models import load_model

  data_env = get_default_data_environment()
  training_env = get_default_training_environment()

  if sys.argv[1] == "train":
    load_default_imdb_data(data_env,
                           max_features=training_env["max-features"],
                           max_length=training_env["max-length"])

    # train model and save
    train_model(training_env, data_env)

    # reload model and eval
    model = load_model(data_env["model-filename"])
    score, acc = model.evaluate(data_env["x-test"],
                                data_env["y-test"],
                                batch_size=training_env["batch-size"])
    print('Test score:', score)
    print('Test accuracy:', acc)
  elif sys.argv[1] == "infer":
    model_filename = sys.argv[2]
    text = sys.argv[3]
    P = infer_with_model(model_filename, text, training_env["max-length"], training_env["max-features"])
    print("prediction: %f" % P)
  else:
    print("ERR: Unknown params: %s" % str(sys.argv))


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D

class KerasLSTMCNNSentiment:
  def __init__(self):
    pass

  def __call__(self, training_env, data_env):
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

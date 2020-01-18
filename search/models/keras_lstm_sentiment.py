"""
A constructor for a simple LSTM to classify sequences
"""
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

class KerasLSTMSentiment:
  def __init__(self):
    pass

  def __call__(self, training_env, data_env):
    #assert "max-features" in model_env
    #assert "sequence-length" in model_env

    model = Sequential()
    model.add(Embedding(data_env["max-features"], data_env["sequence-length"]))
    model.add(LSTM(data_env["sequence-length"], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    return model

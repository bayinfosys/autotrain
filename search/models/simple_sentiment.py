"""
A constructor for a simple network of layers to classify sequences
"""
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

class KerasSimpleSentiment:
  def __init__(self):
    pass

  def __call__(self, training_env, data_env):
    #assert "max-features" in model_env
    #assert "sequence-length" in model_env

    model = Sequential()
    #model.add(Flatten())
    model.add(Dense(data_env["max-length"]))

    for _ in range(training_env["num-layers"]):
      model.add(Dense(training_env["hidden-layer-size"], activation="sigmoid"))
      model.add(Dropout(training_env["dropout-rate"]))

    model.add(Dense(1, activation="sigmoid"))

    return model

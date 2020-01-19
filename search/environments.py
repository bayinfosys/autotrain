import sys

def load_default_imdb_data(data_env, max_features, max_length):
  """loads the default imdba data from keras and
     updates the data_environment dictionary with the arrays
  """
  from keras.preprocessing import sequence
  from keras.datasets import imdb

  print("Loading keras imdb data...")
  (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

  x_train = sequence.pad_sequences(x_train, maxlen=max_length)
  x_test = sequence.pad_sequences(x_test, maxlen=max_length)
  print('x_train shape:', x_train.shape)
  print('x_test shape:', x_test.shape)

  data_env.update({
    "x-train": x_train,
    "y-train": y_train,
    "x-validation": x_test,
    "y-validation": y_test,
    "x-test": x_test,
    "y-test": y_test
  })

def project_text_to_imdb_data(text, max_features, max_length):
  from keras.preprocessing.text import text_to_word_sequence
  from keras.preprocessing import sequence
  from keras.datasets import imdb

  word2index = imdb.get_word_index()

  test = [word2index[word] for word in text_to_word_sequence(text)]

  projected = sequence.pad_sequences([test], maxlen=max_length)
  return projected


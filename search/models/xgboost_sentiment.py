"""
xgboost from python
https://xgboost.readthedocs.io/en/latest/python/python_intro.html
https://xgboost.readthedocs.io/en/latest/python/python_api.html
"""

from environments import (get_default_training_environment,
                          get_default_data_environment,
                          load_default_imdb_data)

class XGBoost:
  def __init__(self):
    pass

  def __call__(self, training_env, data_env):
    from xgboost import XGBClassifier

    assert "learning_rate" in training_env
    #assert "xgboost-depth" in model_env
    #assert "xgboost-num-trees" in model_env

    model = XGBClassifier(
        max_depth=training_env["max_depth"],
        learning_rate=training_env["learning_rate"],
        n_estimators=training_env["n_estimators"],
        silent=False,
        objective='binary:logistic',
        nthread=-1,
        seed=42,
        colsample_bytree=training_env["colsample_bytree"],
        colsample_bylevel=training_env["colsample_bylevel"],
        subsample=training_env["subsample"],
    )
    return model


def train_model(training_env, data_env):

  #assert "x-train" in data_env
  #assert "y-train" in data_env
  #assert "model-filename" in data_env

  model = build_model(training_env)

  print("model: '%s'" % str(model))
  model.fit(data_env["x-train"], data_env["y-train"])

  # save the model
  print("saving model to '%s'" % data_env["model-filename"])
  model.save_model(data_env["model-filename"])
  print("saved model")

  return model

def evaluate_model(data_env):
  import xgboost
  import numpy as np

  #assert "model-filename" in data_env
  #assert "x-test" in data_env

  model = xgboost.XGBClassifier()
  model.load_model(data_env["model-filename"])
  # FIXME: use evals_result
  y_pred = model.predict_proba(data_env["x-test"])[:, 1]

  print(y_pred)
  print(y_pred.shape)

  # compute the score (loss_fn) and accuracy (metric) using the test values
  # FIXME: use the metrics in the parameter set rather than these hardcoded
  #        functions.
  prediction = y_pred.round()
  score = 0.0
  acc = np.sum(data_env["y-test"] == prediction) / data_env["y-test"].shape[0]

  return score, acc

def infer_with_model(data_env):
  assert False

if __name__ == '__main__':
  training_env = get_default_training_environment()
  data_env = get_default_data_environment()

  load_default_imdb_data(data_env,
                         max_features=training_env["max-features"],
                         max_length=training_env["max-length"])

  train_model(training_env, data_env)

  score, acc = evaluate_model(data_env)
  print("test score: %0.8f" % score)
  print("accuracy: %0.8f" % acc)

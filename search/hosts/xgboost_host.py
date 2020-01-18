"""
xgboost training host
https://xgboost.readthedocs.io/en/latest/python/python_intro.html
https://xgboost.readthedocs.io/en/latest/python/python_api.html
"""
def train_model(training_env, data_env, build_fn):
  model = build_fn(training_env, data_env)

  print("model: '%s'" % str(model))
  model.fit(data_env["x-train"],
            data_env["y-train"])

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

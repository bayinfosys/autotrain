"""
xgboost training host
https://xgboost.readthedocs.io/en/latest/python/python_intro.html
https://xgboost.readthedocs.io/en/latest/python/python_api.html

this host uses the tensorboard_logger package to log training and validation
data measurements to tensorboard
https://gist.github.com/tsu-nera/8c37ddf9defca2db46472078351ead33
https://github.com/TeamHG-Memex/tensorboard_logger
FIXME: tensorboard_logger.configure seems to have issues when called multiple times

FIXME: use https://pytorch.org/docs/stable/tensorboard.html instead
"""

from tensorboard_logger import (configure as tb_configure,
                                log_value as tb_log_value)

class XGBoostHost:
  def train_model(training_env, data_env, build_fn):
    def tensorboard_logger(env):
      for k, v in env.evaluation_result_list:
        pos = k.index("-")
        key = k[:pos-2] # NB: xgb usually has validation_0, drop the last 2 chars
        metric = k[pos+1:]
        tb_log_value("%s/%s" % (key, metric), v, step=env.iteration)

    model = build_fn(training_env, data_env)

    print("logging to: '%s'" % data_env["log-dir"])
    tb_configure(data_env["log-dir"])

    print("model: '%s'" % str(model))
    model.fit(data_env["x-train"],
              data_env["y-train"],
              eval_set=[(data_env["x-validation"], data_env["y-validation"])],
              eval_metric=["error", "logloss"],
              early_stopping_rounds=10,
              verbose=False,
              callbacks=[tensorboard_logger])

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

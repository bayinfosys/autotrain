"""
use hyperopt for a model/hyperparameter search

http://hyperopt.github.io/hyperopt/
https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
"""

import logging
import json
import os
from os.path import join
import pickle

from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
from hyperopt import STATUS_OK, STATUS_FAIL

from architects.keras_lstm_sentiment import (train_model as lstm_train_model,
                                             evaluate_model as lstm_evaluate_model)
from architects.keras_lstm_cnn_sentiment import (train_model as lstm_cnn_train_model,
                                                 evaluate_model as lstm_cnn_evaluate_model)
from architects.xgboost_sentiment import (train_model as xgboost_train_model,
                                          evaluate_model as xgboost_evaluate_model)

from environments import (load_default_imdb_data,
                          get_default_data_environment)

from classifier_defs import parse_classifier_file

logger = logging.getLogger(__name__)

def hash_of_dict(D):
  """hashup a dictionary
  """
  from hashlib import md5
  from json import dumps as dict_to_string
  return md5(dict_to_string(sorted(D)).encode("utf-8")).digest().hex()

def fn(params):
  """minimised function
  """
  from time import perf_counter as clock
  from copy import deepcopy

  # get an identifier for this rush
  iterhash = hash_of_dict(params)

  logger.info("DATA-KEYS '%s'" % json.dumps(params))

  # split out the defs
  classifier_type = params["classifier"]["type"]
  classifier_def = params["classifier"]["parameters"]

  # training environment
  training_env = {}
  training_env.update(classifier_def["model-params"])
  training_env.update(classifier_def["training-params"])
  training_env.update(classifier_def["data-params"])

  # get the data params and load the data
  data_env = get_default_data_environment()
  data_env["model-filename"] = join(os.environ["OUTPUT_DIR"], "%s.model" % iterhash)
  data_env.update(classifier_def["data-params"])

  # load the data here because it doesn't like being pickled
  try:
    load_default_imdb_data(data_env,
                           max_features=data_env["max-features"],
                           max_length=data_env["max-length"])
  except FileNotFoundError as es:
    logger.error("training data not found")
    return {"status": STATUS_FAIL,
            "loss": 0.0,
            "exception": str(e)}
  except AssertionError as e:
    logger.error("assert error in load_default_imdb_data exception")
    logger.exception(e)
    return {"status": STATUS_FAIL,
            "loss": 0.0,
            "exception": str(e)}

  # load the appropriate classifier
  if classifier_type == "lstm-cnn":
    train_fn = lstm_cnn_train_model
    eval_fn = lstm_cnn_evaluate_model
  elif classifier_type == "lstm":
    train_fn = lstm_train_model
    eval_fn = lstm_evaluate_model
  elif classifier_type == "xgboost":
    train_fn = xgboost_train_model
    eval_fn = xgboost_evaluate_model

  # fix some errors in the serialisation
  training_env["metrics"] = ["accuracy"]

  try:
    training_time = clock()
    train_fn(training_env, data_env)
    training_time = clock() - training_time
  except ValueError as e:
    logger.error("ValueError exception in train_fn")
    logger.exception(e)
    return {"status": STATUS_FAIL,
            "loss": 0.0,
            "exception": str(e)}
  except AssertionError as e:
    logger.error("AssertionError in training exception")
    logger.exception(e)
    return {"status": STATUS_FAIL,
            "loss": 0.0,
            "exception": str(e)}

  # get the score and accuracy of the model by evaluating
  # againsta test set
  try:
    evaluation_time = clock()
    score, acc = eval_fn(data_env)
    evaluation_time = clock() - evaluation_time
  except ValueError as e:
    logger.error("evaluation exception")
    logger.exception(e)
    return {"status": STATUS_FAIL,
            "loss": 0.0,
            "exception": str(e),
            "training-time": str(training_time)}
  except AssertionError as e:
    logger.error("assert error in evaluation exception")
    logger.exception(e)
    return {"status": STATUS_FAIL,
            "loss": 0.0,
            "exception": str(e)}

  # return success
  return {"status": STATUS_OK,
          "loss": 1.0-acc,
          "score": score,
          "iterhash": iterhash,
          "train-time": str(training_time),
          "evaluation-time": str(evaluation_time)}

if __name__ == "__main__":
  import sys
  import argparse

  ch = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  logger.setLevel(logging.DEBUG)

  parser = argparse.ArgumentParser(description="Model search with hyperopt from a defined set of model types")
  parser.add_argument("-d",
                      "--definitions",
                      default=os.environ.get("CLASSIFIER_DEF"),
                      type=str,
                      help="input model definitions filename")
  parser.add_argument("-m",
                      "--mongodb-conn",
                      default=os.environ.get("MONGO_TRIALS_CONN"),
                      type=str,
                      help="mongo hostname and database for distributed training, e.g.: localhost:1234/<database> the collection 'jobs' will be appended")
  parser.add_argument("-x",
                      "--experiment-name",
                      default=os.environ.get("EXPERIMENT_NAME", "exp1"),
                      type=str,
                      help="experiment name for logging in the database")
  parser.add_argument("-o",
                      "--output-dir",
                      default=".",
                      type=str,
                      help="output directory for pickled trials when not using a mongodb")
  parser.add_argument("--max-evaluations",
                      default=int(os.environ.get("MAX_EVALUATIONS", 100)),
                      type=int,
                      help="maximum evaluations for distributed training")
  args = parser.parse_args()

  for k, v in os.environ.items():
    logger.info("ENV: '%s':'%s'" % (k, v))

  # load the classifiers
  logger.info("loading classifiers...")
  classifiers = parse_classifier_file(args.definitions.strip())

  # test the search space
  # FIXME: make this optional
  import hyperopt.pyll.stochastic
  search_space = [{"type": k, "parameters": classifiers["classifiers"][k]}
                  for k in classifiers["classifiers"]]
  print(json.dumps(hyperopt.pyll.stochastic.sample(search_space), indent=2))

  # stuff
  if args.mongodb_conn is not None:
    logger.info("connecting to '%s'" % args.mongodb_conn)
    # os.environ["MONGODB"] should look like: 'mongo://localhost:1234/foo_db/jobs'
    # http://hyperopt.github.io/hyperopt/scaleout/mongodb/#use-mongotrials
    trials = MongoTrials(args.mongodb_conn,
                         exp_key=args.experiment_name)
    # script entry point is now:
    # > hyperopt-mongo-working --mongo=localhost:1234/foo_db --poll-interval=0.1
  else:
    trials = Trials()

  logger.info("experiment-name: '%s'" % args.experiment_name)

  # do the minimisation
  try:
    best = fmin(
      fn = fn,
      space = {"classifier": hp.choice("classifier", search_space)},
      algo = tpe.suggest,
      max_evals=args.max_evaluations,
      trials=trials)
  except KeyboardInterrupt:
    print("saving trials and exciting...")

  # write out the trials if we are not using mongo
  if isinstance(trials, Trials):
    with open(join(args.output_dir, "%s.pkl" % args.experiment_name), "wb") as trials_f:
      pickle.dump(trials, trials_f, -1)

  # print some output, not sure this is needed
  print (best)

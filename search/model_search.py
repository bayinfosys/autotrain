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
import sys

#logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger("autotrain.model_search")
logger.setLevel(logging.DEBUG)

from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
from hyperopt import STATUS_OK, STATUS_FAIL

from environments import load_default_imdb_data
import classifier_defs # import parse_classifier_pattern


def hash_of_dict(D):
  """hashup a dictionary
  """
  from hashlib import md5
  from json import dumps

  D_s = dumps(D, sort_keys=True).encode("utf-8")
  H_s = md5(D_s).hexdigest()
  return H_s


def fn(params):
  """minimised function
  """
  from time import perf_counter as clock
  from copy import deepcopy

  # get an identifier for this rush
  iterhash = hash_of_dict(deepcopy(params))

  logger.info("DATA-KEYS '%s'" % json.dumps(params))
  logger.info("ITERHASH: '%s'" % iterhash)

  # split out the defs
  classifier_type = params["classifier"]["type"]
  classifier_def = params["classifier"]["parameters"]

  # training environment
  training_env = {}
  training_env.update(classifier_def["model-params"])
  training_env.update(classifier_def["training-params"])
  training_env.update(classifier_def["data-params"])
  training_env.update({"model-type": classifier_type})

  # get the data params and load the data
  data_env = {"model-filename": join(os.environ["OUTPUT_DIR"],
                                     os.getenv("EXPERIMENT_NAME", "experiment"),
                                     "%s.model" % iterhash),
              "log-dir": join(os.environ["OUTPUT_DIR"],
                              "logs",
                              os.getenv("EXPERIMENT_NAME", "experiment"),
                              classifier_type,
                              "%s" % iterhash)
             }
  data_env.update(classifier_def["data-params"])
  logger.info("saving model to: '%s'" % data_env["model-filename"])
  logger.info("saving tblog to: '%s'" % data_env["log-dir"])

  # load the data here because it doesn't like being pickled
  try:
    load_default_imdb_data(data_env,
                           max_features=data_env["max-features"],
                           max_length=data_env["max-length"])
  except Exception as e:
    logger.error("Exception in train_fn")
    logger.exception(e)
    return {"status": STATUS_FAIL,
            "loss": 0.0,
            "exception": str(e)}

  # model builder and training host
  assert "builder" in classifier_def
  assert "hosts" in classifier_def
  from model_loader import load_model_builder, load_hosts
  build_class = load_model_builder(classifier_def["builder"])
  build_fn = build_class()
  train_host, eval_host, _ = load_hosts(classifier_def["hosts"]["training"],
                                        classifier_def["hosts"]["evaluation"],
                                        None)

  # fix some errors in the serialisation
  training_env["metrics"] = ["accuracy"]

  try:
    training_time = clock()
    train_host.train_model(training_env, data_env, build_fn)
    training_time = clock() - training_time
  except Exception as e:
    logger.error("Exception in train_fn")
    logger.exception(e)
    return {"status": STATUS_FAIL,
            "loss": 0.0,
            "exception": str(e)}

  # get the score and accuracy of the model by evaluating
  # againsta test set
  try:
    evaluation_time = clock()
    score, acc = eval_host.evaluate_model(data_env)
    evaluation_time = clock() - evaluation_time
  except Exception as e:
    logger.error("Exception in train_fn")
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
                      help="input model definitions filename pattern")
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
  if args.definitions is not None:
    logger.info("loading classifiers from '%s'" % args.definitions.strip())
    classifiers = classifier_defs.parse_classifier_pattern(args.definitions.strip())
    logger.info("loaded %i classifiers" % len(classifiers))
    print(json.dumps(classifiers, indent=2))
  else:
    logger.error("classifiers not found in CLASSIFIER_DEF or --definitions argument")
    sys.exit()

  # test the search space
  # FIXME: make this optional
  import hyperopt.pyll.stochastic

  search_space = []

  for k in classifiers["classifiers"]:
    logger.info("constructing search space for '%s'" % k)
    search_space.append({"type": k, "parameters": classifiers["classifiers"][k]})
    logger.info(json.dumps(hyperopt.pyll.stochastic.sample(search_space[-1]), indent=2))

  assert len(search_space) > 0, "search_space is empty"

  # stuff
  if args.mongodb_conn is not None:
    logger.info("connecting to '%s'" % args.mongodb_conn)
    # os.environ["MONGO_TRIALS_CONN"] should look like: 'mongo://localhost:1234/foo_db/jobs'
    # http://hyperopt.github.io/hyperopt/scaleout/mongodb/#use-mongotrials
    trials = MongoTrials(args.mongodb_conn,
                         exp_key=args.experiment_name)
    # script entry point is now:
    # > hyperopt-mongo-working --mongo=localhost:1234/foo_db --poll-interval=0.1
  else:
    logger.info("creating local trials")
    trials = Trials()

  print(json.dumps(search_space, indent=2))

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
    out_filename = join(args.output_dir, "%s.pkl" % args.experiment_name)
    logger.info("saving trials to '%s'" % out_filename)

    with open(out_filename, "wb") as trials_f:
      pickle.dump(trials, trials_f, -1)

  # print some output, not sure this is needed
  print (best)

"""dynamic module loading for model builders and training/evaluation/inference hosts

a model builder is a class which returns an object containing parameters which
can be trained by a training host. i.e., a keras model which can be trained by
a keras.model.fit function

a training host is function which takes an object and fits it to data.
an evaluation host is a function which takes an object and evaluates it over data.
an inference host is a function which takes an object and infers results from data.

Hosts are likely to be the same libraries for the same object types, but
may be wrapped up in different ways with different interfaces depending on
usage
"""
import importlib


def load_model_builder(build_classname):
  """loads the method to construct a new instance of this model
  """
  module = importlib.import_module("models")

  class_ = getattr(module, build_classname)
  return class_


def load_hosts(train_classname, eval_classname, infer_classname):
  """loads the hosts of this model process
     all hosts may be the same
  """
  module = importlib.import_module("hosts")

  train_, eval_, infer_ = None, None, None

  if train_classname is not None and len(train_classname) > 0:
    train_ = getattr(module, train_classname)

  if eval_classname is not None and len(eval_classname) > 0:
    eval_ = getattr(module, eval_classname)

  if infer_classname is not None and len(infer_classname) > 0:
    infer_ = getattr(module, infer_classname)

  return train_, eval_, infer_


if __name__ == "__main__":
  import sys
  import json
  from classifier_defs import parse_classifier_file

  # load the classifier definition
  classifier_defs = parse_classifier_file(sys.argv[1])
  classifiers = classifier_defs["classifiers"]

  # get the list of models and hosts
  for k in classifiers:
    print("found: '%s'" % k)
    build = classifiers[k]["builder"]
    train = classifiers[k]["hosts"]["training"]
    eval = classifiers[k]["hosts"]["evaluation"]
    infer = classifiers[k]["hosts"]["inference"]

    print("  build: '%s'" % build)
    print("  train: '%s'" % train)
    print("  eval : '%s'" % eval)
    print("  infer: '%s'" % infer)

    # load the models and hosts
    M = load_model(build)
    T, E, I = load_hosts(train, eval, infer)

    # execute the loaded modules
    print(M)
    print(T)
    print(E)
    print(I)

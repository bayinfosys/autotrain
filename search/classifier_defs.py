"""
parse the classifier definitions file
"""
import logging

logger = logging.getLogger("autotrain.classifier_defs")

def parse_hyperopt_node(n):
  """process a node in a dictionary
     and convert certain entries into hyperopt functions
     ("choice", "uniform", "lognormal")
     return the new dictionary with the correct callable functions
  """
  from hyperopt import hp

  fns = {"choice": lambda n: hp.choice(n["name"], n["choices"]),
         "uniform": lambda n: hp.uniform(n["name"], *n["range"]),
         "lognormal": lambda n: hp.lognormal(n["name"], *n["range"])
        }

  if isinstance(n, type({})) and "type" in n:
    try:
      return fns[n["type"]](n)
    except KeyError:
      raise("KeyNotFound: '%s'" % n["type"])
  elif isinstance(n, str) or isinstance(n, int) or isinstance(n, type([])):
    return n
  else:
    return {k: parse_hyperopt_node(n[k]) for k in n}

def parse_classifier_file(filename):
  """load a definition file of classifiers from <filename>
     and covert the dict entries into callables where appropriate
  """
  if filename.split(".")[-1] == "yaml":
    logger.debug("parsing classifier definition file '%s' as YAML" % filename)
    import yaml
    with open(filename, "r") as f:
      D = yaml.safe_load(f)
  elif filename.split(".")[-1] == "json":
    logger.debug("parsing classifier definition file '%s' as JSON" % filename)

    import json
    with open(filename, "r") as f:
      D = json.load(f)
  else:
    logger.error("unknown classifier definition file format for: '%s'" % filename)
    raise Exception("Unknown filetype: '%s'" % filename.split(".")[-1])

  return parse_hyperopt_node(D)


def parse_classifier_pattern(filename_pattern):
  """glob for a pattern of filenames and process each definition
  """
  from glob import glob

  files = glob(filename_pattern)

  logger.info("found %i model definitions" % len(files))
  print("found %i model definitions" % len(files))
  D = {"classifiers": {}}

  for file in files:
    logger.info("parsing '%s'" % file)
    print("parsing '%s'" % file)

    try:
      df = parse_classifier_file(file)
    except Exception as e:
      logger.error("could not parse '%s'" % file)
      print("could not parse '%s'" % file)
      logger.exception(e)
      continue

    D["classifiers"].update(df["classifiers"])

  logger.info("found classifiers: %s" % ", ".join(D["classifiers"].keys()))
  print("found classifiers: %s" % ", ".join(D["classifiers"].keys()))

  return D


#if __name__ == "__main__":
#  import sys
#  import json
#
#  ch = logging.StreamHandler(sys.stdout)
#  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#  ch.setFormatter(formatter)
#  logger.addHandler(ch)
#  logger.setLevel(logging.DEBUG)
#
#  classifiers = parse_classifier_pattern(sys.argv[1])
#
#  def rprintd(d, l=0):
#    for k in d:
#      print("%s%s" % ("  " * l, k))
#      if isinstance(d[k], dict):
#        rprintd(d[k], l+1)
#
#  rprintd(classifiers)
#
#  # produce an example of the search space from the input
#  # providing this doesn't crash, should be ok?
#  import hyperopt.pyll.stochastic
#  search_space = [{"type": k, "parameters": classifiers[k]} for k in classifiers]
#  print(json.dumps(hyperopt.pyll.stochastic.sample(search_space), indent=2))

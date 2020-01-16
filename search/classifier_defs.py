from hyperopt import hp
import logging

logger = logging.getLogger(__name__)

def parse_hyperopt_node(n):
  """process a node in a dictionary
     and convert certain entries into hyperopt functions
     ("choice", "uniform", "lognormal")
     return the new dictionary with the correct callable functions
  """
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
    logger.info("parsing classifier definition file '%s' as YAML" % filename)
    import yaml
    with open(filename, "r") as f:
      D = yaml.safe_load(f)
  elif filename.split(".")[-1] == "json":
    logger.info("parsing classifier definition file '%s' as JSON" % filename)

    import json
    with open(filename, "r") as f:
      D = json.load(f)
  else:
    logger.error("unknown classifier definition file format for: '%s'" % filename)
    raise Exception("Unknown filetype: '%s'" % filename.split(".")[-1])

  return parse_hyperopt_node(D)

if __name__ == "__main__":
  import sys
  import json

  classifiers = parse_classifier_file(sys.argv[1])

  # produce an example of the search space from the input
  # providing this doesn't crash, should be ok?
  import hyperopt.pyll.stochastic
  search_space = [{"type": k, "parameters": classifiers[k]} for k in classifiers]
  print(json.dumps(hyperopt.pyll.stochastic.sample(search_space), indent=2))

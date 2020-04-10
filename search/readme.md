# Search

Performs the model search by getting the model definitions and variables,
building the search space with [HyperOpt](https://hyperopt.github.io/hyperopt/)
using MongoTrials and setting the workers going over those trials.

# Notes

## HyperOpt version is set at 0.2.2

Hyperopt 0.2.3 introduced a regression with the lower bounded random integer.
(see https://github.com/hyperopt/hyperopt/pull/595).
I haven't been able to isolate the issue yet, but the original defs didn't load.

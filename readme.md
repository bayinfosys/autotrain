# Autotrain
Hyperparameter optimisation in containers.

## Components
+ __docker-compose__ to ochestrate the containers
+ __hyperopt__ to generate the hyperparameters
+ __mongodb__ to hold the hyperopt parameters
+ __tensorboard__ to monitor the training process

## Usage
    ./make.env > .env
    docker-compose -f compose.yaml up --scale model-search 5

to create mongodb username and password and launch five
instances of the model-search process.
tensorboard will be available at __http:://localhost:6006__

# TODO
* GPU training
  * docker-compose does not support --gpus flag, see:
    * https://github.com/NVIDIA/nvidia-docker/issues/1073
    * https://github.com/docker/compose/pull/7124
  * work-arounds are available:
    * https://devblogs.nvidia.com/gpu-containers-runtime/
* kubernetes deployment
  * deploy training onto GPU nodes
* abstract out models and data environment code
  * this stuff is hardcoded at the moment

# Autotrain
Hyperparameter optimisation in containers

## Components
+ __docker-compose__ to orchestrate the containers
+ __hyperopt__ to generate the hyperparameters
+ __mongodb__ to hold the hyperopt parameters
+ __tensorboard__ to monitor the training process

## Usage - docker-compose
    ./make.env > .env
    echo EXPERIMENT_NAME=exp-004 >> .env
    docker-compose -f compose.yaml up --scale model-search 5

To create mongodb username and password, run an experiment named 'exp-004'
with five instances of the model-search process.
Tensorboard will be available at __http:://localhost:6006__

## Usage - minikube

    minikube start
    eval $(minikube -p minikube docker-env)

    docker build -t autotrain.mongo.tester \
                 -f mongo-tester/Dockerfile \
                 ./mongo-tester/

    docker build -t autotrain.search \
                 -f search/Dockerfile \
                 ./search

    kubectl create configmap --from-file ./search/defs/

    kubectl apply -f k8s/config.yaml \
                  -f k8s/mongo.yaml \
                  -f k8s/search-pvc.yaml \
                  -f k8s/search.yaml

monitor stuff with:

    kubectl get nodes,pods,pvc,deployments
                  

## TODO
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

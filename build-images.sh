#!/bin/bash -u

docker build -t autotrain.mongo.tester -f mongo-tester/Dockerfile ./mongo-tester
docker build -t autotrain.search -f search/Dockerfile ./search

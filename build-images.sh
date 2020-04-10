#!/bin/bash -u

docker build -t autotrain.mongo.tester -f mongo-tester/Dockerfile ./mongo-tester
docker build -t autotrain.search -f search/Dockerfile ./search
docker build -t autotrain.data.fetch-s3 -f data/fetch-s3/Dockerfile ./data/fetch-s3
docker build -t autotrain.data.server -f data/server/Dockerfile ./data/server

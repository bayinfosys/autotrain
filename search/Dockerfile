FROM python:3.6-slim-buster

ADD requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    pip install -q \
      --no-cache-dir \
      -r /tmp/requirements.txt

ADD ./*.py /app/
ADD ./models/*.py /app/models/
ADD ./hosts/*.py /app/hosts/

WORKDIR /app

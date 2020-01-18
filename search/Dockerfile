FROM python:3.7-slim-buster

ADD requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    pip install -q \
      --no-cache-dir \
      -r /tmp/requirements.txt

ADD ./*.py /usr/local/bin/
ADD ./architects/*.py /usr/local/bin/architects/

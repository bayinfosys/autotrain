FROM python:3.7-slim-buster

RUN pip install -q --no-cache-dir \
      pymongo

ADD mongo-test-connection.py /usr/local/bin/

CMD ["python", "/usr/local/bin/mongo-test-connection.py"]

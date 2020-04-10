FROM python:3.7-slim-buster

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY fetch-from-csv.py /usr/bin/fetch-from-csv.py

WORKDIR /data

CMD ["python", "/usr/bin/fetch-from-csv.py"]

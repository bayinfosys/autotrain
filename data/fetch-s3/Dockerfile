FROM python:3.7-slim-buster

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /app

WORKDIR /data

CMD ["python", "/app/fetch_from_csv.py", "-h"]

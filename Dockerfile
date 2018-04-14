FROM python:3.6-slim

RUN pip install --trusted-host pypi.python.org -r requirements.txt

WORKDIR .

EXPOSE 80
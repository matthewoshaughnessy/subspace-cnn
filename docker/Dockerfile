FROM python:3.6-slim

WORKDIR /subspace-cnn

ADD . /subspace-cnn

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World

RUN "pwd"

RUN ["python", "./subspace-cnn/cnn_cifar.py"]

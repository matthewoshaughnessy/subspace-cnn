FROM python:3.6-slim

WORKDIR /cnn-torch

ADD . /cnn-torch

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python", "cnn_cifar.py"]

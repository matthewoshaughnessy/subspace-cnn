FROM python:3.6-slim

WORKDIR .

CMD ls

RUN pip install --trusted-host pypi.python.org pytorch
RUN pip install --trusted-host pypi.python.org torchvision

EXPOSE 80

FROM python:3.6-slim

WORKDIR .

CMD ls

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80
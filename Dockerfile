FROM python:3.6-slim

CMD ls

RUN pip install --trusted-host pypi.python.org -r requirements.txt

WORKDIR .

EXPOSE 80
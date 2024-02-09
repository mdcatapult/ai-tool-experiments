FROM --platform=linux/amd64 python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-all

RUN mkdir -p /srv
WORKDIR /srv
COPY requirements.txt /srv
RUN pip3 install -r requirements.txt

COPY src/ /srv

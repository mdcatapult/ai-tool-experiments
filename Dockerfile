FROM --platform=linux/amd64 python:3.9-slim-buster

RUN mkdir -p /srv
WORKDIR /srv
COPY requirements.txt /srv
RUN pip3 install -r requirements.txt

COPY src/ /srv

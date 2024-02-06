FROM --platform=linux/amd64 python:3.9-slim-buster

RUN mkdir -p /srv

ARG OPENAI_API_KEY
ARG DATABASE_USER
ARG DATABASE_PASSWORD
ARG DATABASE_HOST
ARG DATABASE_PORT
ARG DATABASE_NAME
ARG DATABASE_SCHEMA_NAME

WORKDIR /srv
COPY requirements.txt /srv

RUN pip3 install -r requirements.txt

COPY src/ /srv
EXPOSE 8080

CMD ["python3", "-m", "app.main", "--common=app/config.yml"]




# This is installing the pgvector extension for postgres
FROM postgres:14

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-14 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone https://github.com/pgvector/pgvector.git

WORKDIR /tmp/pgvector
RUN make
RUN make install

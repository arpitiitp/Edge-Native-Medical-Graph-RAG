FROM postgres:16

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-16 \
    git \
    make \
    gcc \
    flex \
    bison \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector
WORKDIR /tmp
RUN git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

# Install Apache AGE
RUN git clone -b PG16 https://github.com/apache/age.git && \
    cd age && \
    make USE_PGXS=1 && \
    make USE_PGXS=1 install


# Cleanup
RUN rm -rf /tmp/pgvector /tmp/age

# Set timezone correctly
ENV TZ=UTC

# Add entrypoint script to initially enable extensions if needed
# But standard is to run CREATE EXTENSION inside the DB.

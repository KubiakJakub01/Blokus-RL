# Use an official Python runtime as a base image
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY setup.py ./

RUN pip install -e .[dev]

COPY blokus_rl ./blokus_rl

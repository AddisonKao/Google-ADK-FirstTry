FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY eval/ ./eval/
COPY ingest.py ./ingest.py
COPY docs/ ./docs/

ENV PYTHONUNBUFFERED=1

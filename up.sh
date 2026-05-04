#!/bin/bash
set -e

echo "Starting infra stack (Langfuse, Grafana, Tempo, Prometheus)..."
docker compose -f docker-compose.infra.yml up -d

echo "Waiting for Langfuse to be ready..."
until curl -sf http://localhost:3030/api/public/health > /dev/null 2>&1; do
  sleep 2
done
echo "Langfuse is ready."

echo "Starting app stack (Kafka, agent-consumer, agent-api)..."
docker compose up -d

echo ""
echo "All services started."
echo "  Frontend:   http://localhost:8000"
echo "  Langfuse:   http://localhost:3030"
echo "  Grafana:    http://localhost:3000"
echo "  Prometheus: http://localhost:9090"
echo ""
echo "NOTE: First time setup — run once to populate the insurance knowledge base:"
echo "  docker compose run --rm agent-consumer python ingest.py"

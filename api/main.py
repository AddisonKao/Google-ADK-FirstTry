import json
import os
import threading
import uuid
from pathlib import Path

from confluent_kafka import Consumer, Producer, KafkaError
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import propagate
from opentelemetry.propagators.textmap import DefaultSetter
from pydantic import BaseModel

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = "agent-input"
OUTPUT_TOPIC = "agent-output"

app = FastAPI(title="Kafka ADK Agent API")

# In-memory store: correlation_id → response text
results: dict[str, str] = {}


class KafkaHeaderSetter(DefaultSetter):
    def set(self, carrier: list, key: str, value: str):
        carrier.append((key, value.encode()))


producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})


# Background thread: consume agent-output and populate results store
def _output_consumer():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "agent-api-output-consumer",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([OUTPUT_TOPIC])
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[api output consumer] Error: {msg.error()}")
                continue
            try:
                value = json.loads(msg.value().decode("utf-8"))
                cid = value.get("correlation_id")
                response = value.get("response", "")
                if cid:
                    results[cid] = response
            except Exception as e:
                print(f"[api output consumer] Parse error: {e}")
    finally:
        consumer.close()


@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=_output_consumer, daemon=True)
    t.start()


# ----- Endpoints -----

class InvokeRequest(BaseModel):
    message: str


@app.post("/invoke")
def invoke(req: InvokeRequest):
    correlation_id = str(uuid.uuid4())
    headers: list = []
    propagate.inject(headers, setter=KafkaHeaderSetter())

    producer.produce(
        INPUT_TOPIC,
        key=correlation_id.encode(),
        value=json.dumps({
            "correlation_id": correlation_id,
            "message": req.message,
        }).encode("utf-8"),
        headers=headers,
    )
    producer.flush()
    return {"correlation_id": correlation_id}


@app.get("/result/{correlation_id}")
def get_result(correlation_id: str):
    if correlation_id in results:
        return {"status": "done", "response": results[correlation_id]}
    return JSONResponse(status_code=202, content={"status": "pending"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def frontend():
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text())

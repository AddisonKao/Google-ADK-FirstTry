import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path

from agent.otel_setup import setup as setup_otel
setup_otel()

from confluent_kafka import Consumer, Producer, KafkaError
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from opentelemetry import propagate
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.propagators.textmap import DefaultSetter
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from api.db import init_db, get_connection

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = "agent-input"
OUTPUT_TOPIC = "agent-output"

app = FastAPI(title="Kafka ADK Agent API")
FastAPIInstrumentor.instrument_app(app)

# correlation_id → response text (cleared after SSE delivery)
results: dict[str, str] = {}
# correlation_id → insertion timestamp; entries older than _RESULTS_TTL_SECONDS are evicted
_result_timestamps: dict[str, float] = {}
_RESULTS_TTL_SECONDS = 65  # slightly longer than SSE timeout (60s) so late responses are still served


class KafkaHeaderSetter(DefaultSetter):
    def set(self, carrier: list, key: str, value: str):
        carrier.append((key, value.encode()))


producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})


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
                conversation_id = value.get("conversation_id")
                response = value.get("response", "")

                if cid:
                    results[cid] = response
                    _result_timestamps[cid] = time.time()
                    # Evict stale entries that SSE clients never consumed
                    now = time.time()
                    stale = [k for k, t in _result_timestamps.items() if now - t > _RESULTS_TTL_SECONDS]
                    for k in stale:
                        results.pop(k, None)
                        _result_timestamps.pop(k, None)

                if conversation_id and response is not None:
                    with get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO turns (conversation_id, role, content, correlation_id)"
                                " VALUES (%s, %s, %s, %s)",
                                (conversation_id, "assistant", response, cid),
                            )
                        conn.commit()

            except Exception as e:
                print(f"[api output consumer] Error: {e}")
    finally:
        consumer.close()


@app.on_event("startup")
def startup_event():
    init_db()
    t = threading.Thread(target=_output_consumer, daemon=True)
    t.start()


# ----- Models -----

class TurnRequest(BaseModel):
    message: str


# ----- Endpoints -----

@app.post("/conversations")
def create_conversation():
    conv_id = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO conversations (id) VALUES (%s)", (conv_id,))
        conn.commit()
    return {"conversation_id": conv_id}


@app.post("/conversations/{conversation_id}/turns")
def create_turn(conversation_id: str, req: TurnRequest):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM conversations WHERE id = %s", (conversation_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Conversation not found")

            cur.execute(
                "INSERT INTO turns (conversation_id, role, content) VALUES (%s, %s, %s)",
                (conversation_id, "user", req.message),
            )
        conn.commit()

    correlation_id = str(uuid.uuid4())
    headers: list = []
    propagate.inject(headers, setter=KafkaHeaderSetter())

    producer.produce(
        INPUT_TOPIC,
        key=correlation_id.encode(),
        value=json.dumps({
            "conversation_id": conversation_id,
            "correlation_id": correlation_id,
            "message": req.message,
        }).encode("utf-8"),
        headers=headers,
    )
    producer.flush()
    return {"correlation_id": correlation_id}


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM conversations WHERE id = %s", (conversation_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            cur.execute("DELETE FROM turns WHERE conversation_id = %s", (conversation_id,))
            cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
        conn.commit()


@app.get("/conversations/{conversation_id}/turns")
def get_turns(conversation_id: str):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM conversations WHERE id = %s", (conversation_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Conversation not found")

            cur.execute(
                "SELECT role, content, created_at FROM turns"
                " WHERE conversation_id = %s ORDER BY created_at ASC",
                (conversation_id,),
            )
            rows = cur.fetchall()

    return [
        {"role": r["role"], "content": r["content"], "created_at": r["created_at"].isoformat()}
        for r in rows
    ]


@app.get("/conversations/{conversation_id}/stream/{correlation_id}")
async def stream_result(conversation_id: str, correlation_id: str):
    async def event_generator():
        for _ in range(120):  # 60 seconds at 0.5s intervals
            if correlation_id in results:
                response = results.pop(correlation_id)
                _result_timestamps.pop(correlation_id, None)
                yield f"data: {json.dumps({'response': response})}\n\n"
                return
            await asyncio.sleep(0.5)
        yield f"data: {json.dumps({'error': 'timeout'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "postgres unreachable"})
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def frontend():
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text())

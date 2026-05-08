import json
import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from agent.otel_setup import setup as setup_otel
setup_otel()

from confluent_kafka import Consumer, Producer, KafkaError
from opentelemetry import trace, propagate
from opentelemetry.propagators.textmap import DefaultGetter, DefaultSetter
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.workflow._retry_config import RetryConfig
from google.genai import types as genai_types

from google.adk.agents import Agent
from agent.agent import (
    _model, _DEFAULT_INSTRUCTION, _fetch_instruction,
    _before_model_callback, _after_model_callback, _after_agent_callback,
)
from agent.tools.retrieve import retrieve_tool

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = "agent-input"
OUTPUT_TOPIC = "agent-output"
GROUP_ID = "kafka-adk-consumer"

tracer = trace.get_tracer("kafka_consumer")

# Shared persistent event loop — keeps SQLAlchemy async connection pool alive
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()

session_service = DatabaseSessionService(
    db_url="postgresql+asyncpg://langfuse:langfuse@postgres:5432/adk"
)


def _on_model_error(context, llm_request, error: Exception):
    """Log model errors so we can see rate limits and other failures."""
    print(f"[model error] {type(error).__name__}: {error}")
    return None  # return None to let retry_config handle the retry


def _build_runner() -> Runner:
    """Create a new Agent with the latest prompt from Langfuse (SDK caches 60s by default)."""
    instruction = _fetch_instruction()
    agent = Agent(
        name="kafka_agent",
        model=_model,
        instruction=instruction,
        tools=[retrieve_tool],
        before_model_callback=_before_model_callback,
        after_model_callback=_after_model_callback,
        after_agent_callback=_after_agent_callback,
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            backoff_factor=2.0,
        ),
        on_model_error_callback=_on_model_error,
    )
    return Runner(agent=agent, session_service=session_service, app_name="kafka_agent")

producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})


class KafkaHeaderGetter(DefaultGetter):
    def get(self, carrier: list, key: str):
        for k, v in carrier:
            if k == key:
                return [v.decode() if isinstance(v, bytes) else v]
        return []

    def keys(self, carrier: list):
        return [k for k, _ in carrier]


class KafkaHeaderSetter(DefaultSetter):
    def set(self, carrier: list, key: str, value: str):
        carrier.append((key, value.encode()))


async def process_message(message_value: dict, headers: list) -> tuple[str, list]:
    """Run the ADK agent and return (response_text, out_headers)."""
    ctx = propagate.extract(headers, getter=KafkaHeaderGetter())

    out_headers = []
    with tracer.start_as_current_span("kafka_consumer.process", context=ctx):
        correlation_id = message_value.get("correlation_id", "unknown")
        conversation_id = message_value.get("conversation_id", "unknown")
        user_text = message_value.get("message", "")

        # Use conversation_id as session_id — deterministic, survives consumer restarts
        session = await session_service.get_session(
            app_name="kafka_agent",
            user_id="kafka-user",
            session_id=conversation_id,
        )
        if session is None:
            session = await session_service.create_session(
                app_name="kafka_agent",
                user_id="kafka-user",
                session_id=conversation_id,
            )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_text)],
        )

        # Run agent once; RetryConfig handles LLM-level retries, outer
        # process_message_sync handles transient infrastructure retries.
        runner = _build_runner()
        response_text = ""
        try:
            async for event in runner.run_async(
                user_id="kafka-user",
                session_id=conversation_id,
                new_message=content,
            ):
                if event.is_final_response():
                    parts = (event.content.parts or []) if event.content else []
                    for part in parts:
                        text = getattr(part, "text", None)
                        if text and text.strip():
                            response_text = text
                    if not response_text:
                        actions = getattr(event, "actions", None)
                        if actions and getattr(actions, "skip_summarization", False):
                            fn_responses = event.get_function_responses()
                            if fn_responses:
                                result = fn_responses[0].response
                                response_text = result.get("result", str(result)) if isinstance(result, dict) else str(result or "")
        except Exception as _runner_exc:
            print(f"[consumer] runner exception ({type(_runner_exc).__name__}): {_runner_exc}")
            raise  # re-raise so outer transient-error retry in process_message_sync can handle it

        # Inject trace context while span is still active
        propagate.inject(out_headers, setter=KafkaHeaderSetter())

    return response_text, out_headers


MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 7]  # seconds between retries


def process_message_sync(value: dict, headers: list):
    """Dispatch to the shared persistent event loop. Retries on transient errors."""
    correlation_id = value.get("correlation_id", "unknown")
    conversation_id = value.get("conversation_id", "unknown")

    response_text = None
    out_headers = []

    for attempt in range(MAX_RETRIES):
        try:
            future = asyncio.run_coroutine_threadsafe(process_message(value, headers), _loop)
            response_text, out_headers = future.result(timeout=120)
            break
        except Exception as e:
            is_transient = any(code in str(e) for code in ["503", "429", "UNAVAILABLE", "Resource exhausted", "name resolution", "gaierror", "ClientConnector"])
            if is_transient and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"[consumer] Transient error (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s: {e}")
                import time; time.sleep(delay)
            else:
                print(f"[consumer] Failed after {attempt + 1} attempt(s): {e}")
                response_text = "抱歉，AI 服務目前暫時無法使用，請稍後再試。"
                break

    producer.produce(
        OUTPUT_TOPIC,
        key=correlation_id.encode(),
        value=json.dumps({
            "correlation_id": correlation_id,
            "conversation_id": conversation_id,
            "response": response_text,
        }).encode("utf-8"),
        headers=out_headers,
    )
    producer.flush()
    print(f"[consumer] Published response for {correlation_id}")


def run_consumer():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
    })

    executor = ThreadPoolExecutor(max_workers=3)
    consumer.subscribe([INPUT_TOPIC])
    print(f"[consumer] Subscribed to {INPUT_TOPIC}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[consumer] Error: {msg.error()}")
                continue

            try:
                value = json.loads(msg.value().decode("utf-8"))
                headers = msg.headers() or []
                correlation_id = value.get("correlation_id", "unknown")
                print(f"[consumer] Dispatching message {correlation_id}")
                executor.submit(process_message_sync, value, headers)

            except Exception as e:
                print(f"[consumer] Failed to dispatch message: {e}")

    finally:
        consumer.close()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    run_consumer()

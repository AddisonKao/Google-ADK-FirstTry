import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from agent.otel_setup import setup as setup_otel
setup_otel()

from confluent_kafka import Consumer, Producer, KafkaError
from opentelemetry import trace, propagate
from opentelemetry.propagators.textmap import DefaultGetter, DefaultSetter
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from google.adk.agents import Agent
from agent.agent import (
    _model, _DEFAULT_INSTRUCTION, _fetch_instruction,
    echo_tool, _before_model_callback, _after_model_callback, _after_agent_callback,
)
from agent.tools.retrieve import retrieve_tool

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = "agent-input"
OUTPUT_TOPIC = "agent-output"
GROUP_ID = "kafka-adk-consumer"

tracer = trace.get_tracer("kafka_consumer")

# session_service 保留在 module level，確保 conversation 狀態不會跨 invocation 消失
session_service = InMemorySessionService()


def _build_runner() -> Runner:
    """Create a new Agent with the latest prompt from Langfuse (SDK caches 60s by default)."""
    instruction = _fetch_instruction()
    agent = Agent(
        name="kafka_agent",
        model=_model,
        instruction=instruction,
        tools=[echo_tool, retrieve_tool],
        before_model_callback=_before_model_callback,
        after_model_callback=_after_model_callback,
        after_agent_callback=_after_agent_callback,
    )
    return Runner(agent=agent, session_service=session_service, app_name="kafka_agent")

# conversation_id → ADK session_id
sessions: dict[str, str] = {}

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

        # Reuse existing session for this conversation, or create a new one
        if conversation_id in sessions:
            session = await session_service.get_session(
                app_name="kafka_agent",
                user_id="kafka-user",
                session_id=sessions[conversation_id],
            )
            if session is None:
                session = await session_service.create_session(
                    app_name="kafka_agent", user_id="kafka-user"
                )
                sessions[conversation_id] = session.id
        else:
            session = await session_service.create_session(
                app_name="kafka_agent", user_id="kafka-user"
            )
            sessions[conversation_id] = session.id

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_text)],
        )

        runner = _build_runner()
        response_text = ""
        async for event in runner.run_async(
            user_id="kafka-user",
            session_id=session.id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text

        # Inject trace context while span is still active
        propagate.inject(out_headers, setter=KafkaHeaderSetter())

    return response_text, out_headers


def process_message_sync(value: dict, headers: list):
    """Called in thread pool. Each thread gets its own event loop."""
    correlation_id = value.get("correlation_id", "unknown")
    conversation_id = value.get("conversation_id", "unknown")

    try:
        response_text, out_headers = asyncio.run(process_message(value, headers))

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

    except Exception as e:
        print(f"[consumer] Failed to process message {correlation_id}: {e}")


def run_consumer():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
    })

    executor = ThreadPoolExecutor(max_workers=5)
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

import json
import os
import asyncio
from agent.otel_setup import setup as setup_otel
setup_otel()

from confluent_kafka import Consumer, Producer, KafkaError
from opentelemetry import trace, propagate
from opentelemetry.propagators.textmap import DefaultGetter, DefaultSetter
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agent.agent import root_agent

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = "agent-input"
OUTPUT_TOPIC = "agent-output"
GROUP_ID = "kafka-adk-consumer"

tracer = trace.get_tracer("kafka_consumer")

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="kafka_agent")


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


async def process_message(message_value: dict, headers: list) -> str:
    """Run the ADK agent and return the final text response."""
    ctx = propagate.extract(headers, getter=KafkaHeaderGetter())

    with tracer.start_as_current_span("kafka_consumer.process", context=ctx):
        correlation_id = message_value.get("correlation_id", "unknown")
        user_text = message_value.get("message", "")

        session = await session_service.create_session(
            app_name="kafka_agent",
            user_id="kafka-user",
        )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_text)],
        )

        response_text = ""
        async for event in runner.run_async(
            user_id="kafka-user",
            session_id=session.id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text

        return response_text


def run_consumer():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
    })
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

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

                print(f"[consumer] Processing message {correlation_id}")
                response_text = asyncio.run(process_message(value, headers))

                out_headers = []
                propagate.inject(out_headers, setter=KafkaHeaderSetter())

                producer.produce(
                    OUTPUT_TOPIC,
                    key=correlation_id.encode(),
                    value=json.dumps({
                        "correlation_id": correlation_id,
                        "response": response_text,
                    }).encode("utf-8"),
                    headers=out_headers,
                )
                producer.flush()
                print(f"[consumer] Published response for {correlation_id}")

            except Exception as e:
                print(f"[consumer] Failed to process message: {e}")

    finally:
        consumer.close()


if __name__ == "__main__":
    run_consumer()

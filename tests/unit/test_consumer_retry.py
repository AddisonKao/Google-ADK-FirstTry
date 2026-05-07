"""Unit tests for consumer retry logic."""
import pytest
from unittest.mock import patch, MagicMock


def test_transient_503_triggers_retry():
    """503 error should trigger retries, not immediate failure."""
    call_count = {"n": 0}

    def failing_process(*args, **kwargs):
        call_count["n"] += 1
        raise Exception("503 UNAVAILABLE service temporarily unavailable")

    mock_future = MagicMock()
    mock_future.result.side_effect = failing_process

    with patch("agent.consumer.asyncio.run_coroutine_threadsafe", return_value=mock_future), \
         patch("agent.consumer.producer") as mock_producer, \
         patch("time.sleep"):  # skip actual sleep in tests

        import agent.consumer as consumer_mod
        consumer_mod.process_message_sync(
            {"correlation_id": "test-id", "conversation_id": "conv-1", "message": "hi"},
            []
        )

    assert call_count["n"] == consumer_mod.MAX_RETRIES


def test_final_failure_publishes_error_message():
    """After all retries fail, should publish Chinese error message."""
    def always_fail(*args, **kwargs):
        raise Exception("503 UNAVAILABLE")

    mock_future = MagicMock()
    mock_future.result.side_effect = always_fail

    published_value = {}

    def mock_produce(*args, **kwargs):
        import json
        published_value["data"] = json.loads(kwargs.get("value", b"{}").decode())

    with patch("agent.consumer.asyncio.run_coroutine_threadsafe", return_value=mock_future), \
         patch("agent.consumer.producer") as mock_producer, \
         patch("time.sleep"):

        mock_producer.produce.side_effect = mock_produce
        mock_producer.flush = MagicMock()

        import agent.consumer as consumer_mod
        consumer_mod.process_message_sync(
            {"correlation_id": "test-id", "conversation_id": "conv-1", "message": "hi"},
            []
        )

    response = published_value.get("data", {}).get("response", "")
    assert "抱歉" in response or "無法使用" in response


def test_non_transient_error_does_not_retry():
    """Non-503/429 errors should not retry (only 1 attempt)."""
    call_count = {"n": 0}

    def non_transient_fail(*args, **kwargs):
        call_count["n"] += 1
        raise ValueError("Some programming error — not transient")

    mock_future = MagicMock()
    mock_future.result.side_effect = non_transient_fail

    with patch("agent.consumer.asyncio.run_coroutine_threadsafe", return_value=mock_future), \
         patch("agent.consumer.producer") as mock_producer, \
         patch("time.sleep"):
        mock_producer.produce = MagicMock()
        mock_producer.flush = MagicMock()

        import agent.consumer as consumer_mod
        consumer_mod.process_message_sync(
            {"correlation_id": "test-id", "conversation_id": "conv-1", "message": "hi"},
            []
        )

    assert call_count["n"] == 1  # no retry for non-transient

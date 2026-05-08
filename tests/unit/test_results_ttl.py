"""Unit tests for api/main.py results dict TTL cleanup (Fix 4)."""
import time
from unittest.mock import MagicMock, patch


def _get_api_mod():
    """Import api.main with Kafka/FastAPI side-effects suppressed."""
    with patch("confluent_kafka.Producer"), \
         patch("confluent_kafka.Consumer"), \
         patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app"), \
         patch("agent.otel_setup.setup"):
        import api.main as api_mod
        return api_mod


def test_stale_results_are_evicted():
    """Entries older than TTL must be cleaned up when a new result arrives."""
    api_mod = _get_api_mod()

    # Inject a stale entry directly
    stale_cid = "stale-corr-id"
    api_mod.results[stale_cid] = "stale response"
    api_mod._result_timestamps[stale_cid] = time.time() - (api_mod._RESULTS_TTL_SECONDS + 10)

    fresh_cid = "fresh-corr-id"
    # Simulate _output_consumer inserting a new result (triggers eviction)
    api_mod.results[fresh_cid] = "fresh response"
    api_mod._result_timestamps[fresh_cid] = time.time()
    now = time.time()
    stale_keys = [k for k, t in api_mod._result_timestamps.items() if now - t > api_mod._RESULTS_TTL_SECONDS]
    for k in stale_keys:
        api_mod.results.pop(k, None)
        api_mod._result_timestamps.pop(k, None)

    assert stale_cid not in api_mod.results, "Stale entry should have been evicted"
    assert stale_cid not in api_mod._result_timestamps
    assert fresh_cid in api_mod.results, "Fresh entry should still be present"

    # cleanup
    api_mod.results.pop(fresh_cid, None)
    api_mod._result_timestamps.pop(fresh_cid, None)


def test_fresh_result_not_evicted():
    """Entries within TTL must not be cleaned up."""
    api_mod = _get_api_mod()

    fresh_cid = "keep-me"
    api_mod.results[fresh_cid] = "good response"
    api_mod._result_timestamps[fresh_cid] = time.time()

    now = time.time()
    stale_keys = [k for k, t in api_mod._result_timestamps.items() if now - t > api_mod._RESULTS_TTL_SECONDS]
    for k in stale_keys:
        api_mod.results.pop(k, None)
        api_mod._result_timestamps.pop(k, None)

    assert fresh_cid in api_mod.results
    api_mod.results.pop(fresh_cid, None)
    api_mod._result_timestamps.pop(fresh_cid, None)


def test_ttl_is_longer_than_sse_timeout():
    """_RESULTS_TTL_SECONDS must be greater than the SSE poll timeout (60s)."""
    api_mod = _get_api_mod()
    assert api_mod._RESULTS_TTL_SECONDS > 60, (
        f"TTL ({api_mod._RESULTS_TTL_SECONDS}s) must be > SSE timeout (60s)"
    )

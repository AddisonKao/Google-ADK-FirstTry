"""Unit tests for agent.py callback fixes: pending_traces memory leak and model name."""
import time
from unittest.mock import MagicMock, patch


def _make_ctx(invocation_id="test-inv-1"):
    ctx = MagicMock()
    ctx.invocation_id = invocation_id
    return ctx


# ── Fix 1: _pending_traces memory leak ────────────────────────────────────────

def test_after_agent_callback_removes_entry():
    """_after_agent_callback must clean up the entry from _pending_traces."""
    import agent.agent as agent_mod

    agent_mod._pending_traces["inv-cleanup"] = {
        "trace": MagicMock(),
        "gen_count": 0,
        "inserted_at": time.time(),
    }
    assert "inv-cleanup" in agent_mod._pending_traces

    ctx = _make_ctx("inv-cleanup")
    agent_mod._after_agent_callback(ctx)
    assert "inv-cleanup" not in agent_mod._pending_traces


def test_evict_stale_traces_removes_old_entries():
    """_evict_stale_traces must remove entries older than TTL."""
    import agent.agent as agent_mod

    old_time = time.time() - (agent_mod._PENDING_TRACES_TTL_SECONDS + 60)
    agent_mod._pending_traces["stale-inv"] = {
        "trace": MagicMock(),
        "gen_count": 0,
        "inserted_at": old_time,
    }
    agent_mod._pending_traces["fresh-inv"] = {
        "trace": MagicMock(),
        "gen_count": 0,
        "inserted_at": time.time(),
    }

    agent_mod._evict_stale_traces()

    assert "stale-inv" not in agent_mod._pending_traces
    assert "fresh-inv" in agent_mod._pending_traces

    # cleanup
    agent_mod._pending_traces.pop("fresh-inv", None)


def test_evict_stale_traces_hard_cap():
    """When dict exceeds MAXLEN, oldest entries are dropped."""
    import agent.agent as agent_mod

    original = dict(agent_mod._pending_traces)
    agent_mod._pending_traces.clear()

    try:
        # Fill to MAXLEN
        for i in range(agent_mod._PENDING_TRACES_MAXLEN):
            agent_mod._pending_traces[f"inv-{i}"] = {
                "trace": MagicMock(),
                "gen_count": 0,
                "inserted_at": time.time(),
            }
        assert len(agent_mod._pending_traces) == agent_mod._PENDING_TRACES_MAXLEN

        agent_mod._evict_stale_traces()
        # All fresh, TTL eviction removes nothing, but hard cap trims to < MAXLEN
        assert len(agent_mod._pending_traces) < agent_mod._PENDING_TRACES_MAXLEN
    finally:
        agent_mod._pending_traces.clear()
        agent_mod._pending_traces.update(original)


# ── Fix 5: correct model name in Langfuse generation ──────────────────────────

def test_model_name_uses_gemini_model_env(monkeypatch):
    """When no OpenAI env vars, model_name should use _gemini_model not hardcoded string."""
    import agent.agent as agent_mod

    monkeypatch.setattr(agent_mod, "_openai_base", None)
    monkeypatch.setattr(agent_mod, "_openai_key", None)
    monkeypatch.setattr(agent_mod, "_gemini_model", "gemini-2.5-flash-lite")

    invocation_id = "inv-model-test"
    mock_trace = MagicMock()
    agent_mod._pending_traces[invocation_id] = {
        "trace": mock_trace,
        "gen_count": 0,
        "inserted_at": time.time(),
        "gen_input": "test input",
        "gen_start_time": MagicMock(),
    }

    mock_lf = MagicMock()
    monkeypatch.setattr(agent_mod, "_lf", mock_lf)
    monkeypatch.setattr(agent_mod, "_get_langfuse", lambda: mock_lf)

    mock_response = MagicMock()
    mock_response.content = None
    mock_response.usage_metadata = None

    ctx = _make_ctx(invocation_id)
    agent_mod._after_model_callback(ctx, mock_response)

    # Verify generation was called with gemini-2.5-flash-lite, not hardcoded gemini-2.0-flash
    call_kwargs = mock_trace.generation.call_args[1]
    assert call_kwargs["model"] == "gemini-2.5-flash-lite", (
        f"Expected gemini-2.5-flash-lite but got {call_kwargs['model']!r}"
    )

    agent_mod._pending_traces.pop(invocation_id, None)


def test_model_name_uses_openai_model_when_configured(monkeypatch):
    """When OpenAI env vars are set, model_name should use _openai_model."""
    import agent.agent as agent_mod

    monkeypatch.setattr(agent_mod, "_openai_base", "https://api.example.com/v1")
    monkeypatch.setattr(agent_mod, "_openai_key", "sk-test")
    monkeypatch.setattr(agent_mod, "_openai_model", "gpt-4o-mini")

    invocation_id = "inv-openai-test"
    mock_trace = MagicMock()
    agent_mod._pending_traces[invocation_id] = {
        "trace": mock_trace,
        "gen_count": 0,
        "inserted_at": time.time(),
        "gen_input": "test input",
        "gen_start_time": MagicMock(),
    }

    mock_lf = MagicMock()
    monkeypatch.setattr(agent_mod, "_lf", mock_lf)
    monkeypatch.setattr(agent_mod, "_get_langfuse", lambda: mock_lf)

    mock_response = MagicMock()
    mock_response.content = None
    mock_response.usage_metadata = None

    ctx = _make_ctx(invocation_id)
    agent_mod._after_model_callback(ctx, mock_response)

    call_kwargs = mock_trace.generation.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"

    agent_mod._pending_traces.pop(invocation_id, None)

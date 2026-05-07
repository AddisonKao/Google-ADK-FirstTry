"""Unit tests for embed mode selection — no real API calls."""
import pytest


def test_gemini_mode_selected_when_no_api_base(monkeypatch):
    """Without EMBEDDING_API_BASE, should call _embed_gemini."""
    monkeypatch.delenv("EMBEDDING_API_BASE", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    called = {"gemini": False, "openai": False}

    import agent.tools.embed as embed_mod
    monkeypatch.setattr(embed_mod, "EMBEDDING_API_BASE", "")
    monkeypatch.setattr(embed_mod, "EMBEDDING_API_KEY", "")
    monkeypatch.setattr(embed_mod, "_embed_gemini", lambda texts: (called.__setitem__("gemini", True) or [[0.0] * 768]))
    monkeypatch.setattr(embed_mod, "_embed_openai_compat", lambda texts: (called.__setitem__("openai", True) or [[0.0] * 768]))

    embed_mod.embed(["test text"])
    assert called["gemini"] is True
    assert called["openai"] is False


def test_openai_compat_mode_selected_when_api_base_set(monkeypatch):
    """With EMBEDDING_API_BASE + EMBEDDING_API_KEY, should call _embed_openai_compat."""
    import agent.tools.embed as embed_mod
    monkeypatch.setattr(embed_mod, "EMBEDDING_API_BASE", "https://api.example.com/v1")
    monkeypatch.setattr(embed_mod, "EMBEDDING_API_KEY", "sk-test-key")

    called = {"gemini": False, "openai": False}
    monkeypatch.setattr(embed_mod, "_embed_gemini", lambda texts: (called.__setitem__("gemini", True) or [[0.0] * 768]))
    monkeypatch.setattr(embed_mod, "_embed_openai_compat", lambda texts: (called.__setitem__("openai", True) or [[0.0] * 768]))

    embed_mod.embed(["test text"])
    assert called["openai"] is True
    assert called["gemini"] is False


def test_embed_one_returns_single_vector(monkeypatch):
    """embed_one should return a single flat list."""
    import agent.tools.embed as embed_mod
    monkeypatch.setattr(embed_mod, "_embed_gemini", lambda texts: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(embed_mod, "EMBEDDING_API_BASE", "")
    monkeypatch.setattr(embed_mod, "EMBEDDING_API_KEY", "")

    result = embed_mod.embed_one("single text")
    assert isinstance(result, list)
    assert result[0] == 0.1

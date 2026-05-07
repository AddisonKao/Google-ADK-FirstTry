"""Unit tests for retrieve_tool — no real DB connection needed."""
import pytest


def test_retrieve_returns_no_results_message(monkeypatch):
    """Empty pgvector result → friendly message."""
    import psycopg2

    class MockCursor:
        def execute(self, *args): pass
        def fetchall(self): return []
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class MockConn:
        def cursor(self): return MockCursor()
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr(psycopg2, "connect", lambda *a, **kw: MockConn())

    # Also mock embed_one to avoid real API call
    import agent.tools.retrieve as retrieve_mod
    monkeypatch.setattr(retrieve_mod, "embed_one", lambda q: [0.0] * 768)

    from agent.tools.retrieve import retrieve_tool
    result = retrieve_tool("什麼是保險？")
    assert "No relevant information" in result


def test_retrieve_returns_db_unavailable_on_exception(monkeypatch):
    """DB connection failure → knowledge base unavailable message."""
    import psycopg2
    monkeypatch.setattr(psycopg2, "connect", lambda *a, **kw: (_ for _ in ()).throw(Exception("connection refused")))

    import agent.tools.retrieve as retrieve_mod
    monkeypatch.setattr(retrieve_mod, "embed_one", lambda q: [0.0] * 768)

    from agent.tools.retrieve import retrieve_tool
    result = retrieve_tool("測試")
    assert "Knowledge base unavailable" in result or "unavailable" in result.lower()


def test_retrieve_returns_chunks_when_rows_exist(monkeypatch):
    """When rows exist, return concatenated content."""
    import psycopg2

    class MockCursor:
        def execute(self, *args): pass
        def fetchall(self): return [("chunk1 content", 0.9), ("chunk2 content", 0.8)]
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class MockConn:
        def cursor(self): return MockCursor()
        def close(self): pass

    monkeypatch.setattr(psycopg2, "connect", lambda *a, **kw: MockConn())

    import agent.tools.retrieve as retrieve_mod
    monkeypatch.setattr(retrieve_mod, "embed_one", lambda q: [0.0] * 768)

    from agent.tools.retrieve import retrieve_tool
    result = retrieve_tool("測試")
    assert "chunk1 content" in result
    assert "chunk2 content" in result

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
    assert "No relevant information" in result.get("result", "")


def test_retrieve_returns_db_unavailable_on_exception(monkeypatch):
    """DB connection failure → knowledge base unavailable message."""
    import psycopg2
    monkeypatch.setattr(psycopg2, "connect", lambda *a, **kw: (_ for _ in ()).throw(Exception("connection refused")))

    import agent.tools.retrieve as retrieve_mod
    monkeypatch.setattr(retrieve_mod, "embed_one", lambda q: [0.0] * 768)

    from agent.tools.retrieve import retrieve_tool
    result = retrieve_tool("測試")
    result_text = result.get("result", "") if isinstance(result, dict) else str(result)
    assert "Knowledge base unavailable" in result_text or "unavailable" in result_text.lower()


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
    result_text = result.get("result", "") if isinstance(result, dict) else str(result)
    assert "chunk1 content" in result_text
    assert "chunk2 content" in result_text


def test_retrieve_closes_connection_on_cursor_exception(monkeypatch):
    """conn.close() must be called even if cursor.execute() raises (Fix 2: connection leak)."""
    import psycopg2

    close_called = {"n": 0}

    class MockCursor:
        def execute(self, *args):
            raise RuntimeError("simulated cursor error")
        def fetchall(self): return []
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class MockConn:
        def cursor(self): return MockCursor()
        def close(self): close_called["n"] += 1

    monkeypatch.setattr(psycopg2, "connect", lambda *a, **kw: MockConn())

    import agent.tools.retrieve as retrieve_mod
    monkeypatch.setattr(retrieve_mod, "embed_one", lambda q: [0.0] * 768)

    from agent.tools.retrieve import retrieve_tool
    result = retrieve_tool("測試")
    # Should not crash, and close must have been called
    assert close_called["n"] == 1, "conn.close() was not called after cursor exception"
    result_text = result.get("result", "") if isinstance(result, dict) else str(result)
    assert "unavailable" in result_text.lower()


def test_retrieve_sql_uses_vector_search_constant(monkeypatch):
    """VECTOR_SEARCH_SQL constant must be used (Fix 6: HYBRID_SQL renamed)."""
    import agent.tools.retrieve as retrieve_mod
    assert hasattr(retrieve_mod, "VECTOR_SEARCH_SQL"), "VECTOR_SEARCH_SQL constant not found"
    assert not hasattr(retrieve_mod, "HYBRID_SQL"), "HYBRID_SQL should have been removed"

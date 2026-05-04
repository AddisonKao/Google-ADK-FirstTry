"""RAG retrieve tool for ADK agent. Hybrid search: vector + BM25 (tsvector)."""
import os

import psycopg2

from agent.tools.embed import embed_one

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://langfuse:langfuse@localhost:5432/langfuse",
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

HYBRID_SQL = """
SELECT content,
    0.6 * (1 - (embedding <=> %s::vector)) +
    0.4 * ts_rank(content_tsv, plainto_tsquery('english', %s)) AS score
FROM rag.documents
WHERE content_tsv @@ plainto_tsquery('english', %s)
   OR (embedding <=> %s::vector) < 0.5
ORDER BY score DESC
LIMIT %s;
"""


def retrieve_tool(query: str) -> str:
    """Search the insurance knowledge base and return relevant context.

    Args:
        query: The question or topic to search for in the knowledge base.

    Returns:
        A string containing the most relevant passages from the knowledge base,
        separated by '---'. Returns a message if no results found.
    """
    try:
        query_vec = embed_one(query)
        vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"

        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(HYBRID_SQL, (vec_str, query, query, vec_str, RAG_TOP_K))
            rows = cur.fetchall()
        conn.close()

        if not rows:
            return "No relevant information found in the knowledge base."

        chunks = [row[0] for row in rows]
        return "\n---\n".join(chunks)

    except Exception as e:
        return f"Knowledge base unavailable: {e}"

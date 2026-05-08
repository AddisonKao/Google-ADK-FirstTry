"""RAG retrieve tool for ADK agent. Vector similarity search via pgvector."""
import os

import psycopg2

from agent.tools.embed import embed_one

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://langfuse:langfuse@localhost:5432/langfuse",
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

VECTOR_SEARCH_SQL = """
SELECT content,
    1 - (embedding <=> %s::vector) AS score
FROM rag.documents
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


def retrieve_tool(query: str) -> str:
    """Search the insurance knowledge base and return relevant context.

    Args:
        query: The question or topic to search for in the knowledge base.

    Returns:
        A dict with key 'result' containing relevant passages, or an error message.
    """
    try:
        query_vec = embed_one(query)
        vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"

        conn = psycopg2.connect(DATABASE_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(VECTOR_SEARCH_SQL, (vec_str, vec_str, RAG_TOP_K))
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return {"result": "No relevant information found in the knowledge base."}

        chunks = [row[0] for row in rows]
        return {"result": "\n---\n".join(chunks)}

    except Exception as e:
        return {"result": f"Knowledge base unavailable: {e}"}

"""
One-time script to ingest InsuranceQA-v2 into pgvector.

Usage:
    python ingest.py

Env vars (from .env):
    GEMINI_API_KEY or (EMBEDDING_API_BASE + EMBEDDING_API_KEY)
    EMBEDDING_MODEL    default: text-embedding-004
    DATABASE_URL       default: postgresql://langfuse:langfuse@localhost:5432/langfuse
"""
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from agent.tools.embed import embed

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://langfuse:langfuse@localhost:5432/langfuse",
)
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "100"))
MAX_RECORDS = int(os.getenv("INGEST_MAX_RECORDS", "0"))  # 0 = all


def init_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS rag;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.documents (
                id          SERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                embedding   vector(768),
                content_tsv TSVECTOR,
                source      TEXT DEFAULT 'insuranceqa-v2',
                metadata    JSONB DEFAULT '{}'
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS rag_documents_tsv_idx
            ON rag.documents USING GIN(content_tsv);
        """)
    conn.commit()
    print("[ingest] Schema and table ready.")


def truncate(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE rag.documents RESTART IDENTITY;")
    conn.commit()
    print("[ingest] Table truncated.")


def load_answers() -> list[str]:
    print("[ingest] Downloading deccan-ai/insuranceQA-v2 ...")
    ds = load_dataset("deccan-ai/insuranceQA-v2", split="train")
    answers = []
    for row in ds:
        answer = row.get("output") or row.get("ground_truth") or row.get("answer")
        if answer:
            answers.append(str(answer))
    # Deduplicate
    answers = list(dict.fromkeys(answers))
    if MAX_RECORDS > 0:
        answers = answers[:MAX_RECORDS]
        print(f"[ingest] Limiting to {MAX_RECORDS} records (INGEST_MAX_RECORDS).")
    print(f"[ingest] Loaded {len(answers)} unique answers.")
    return answers


def insert_batch(conn, texts: list[str], vectors: list[list[float]]):
    with conn.cursor() as cur:
        for text, vec in zip(texts, vectors):
            cur.execute(
                """
                INSERT INTO rag.documents (content, embedding, content_tsv)
                VALUES (%s, %s, to_tsvector('english', %s))
                """,
                (text, vec, text),
            )
    conn.commit()


def main():
    conn = psycopg2.connect(DATABASE_URL)
    init_schema(conn)
    truncate(conn)

    answers = load_answers()
    total = len(answers)
    inserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = answers[i : i + BATCH_SIZE]
        print(f"[ingest] Embedding batch {i // BATCH_SIZE + 1} / {(total + BATCH_SIZE - 1) // BATCH_SIZE} ...")
        vectors = embed(batch)
        insert_batch(conn, batch, vectors)
        inserted += len(batch)
        print(f"[ingest] {inserted}/{total} records inserted.")

    conn.close()
    print(f"[ingest] Done. {inserted} documents ingested into rag.documents.")


if __name__ == "__main__":
    main()

"""
One-time script to ingest local Markdown docs into pgvector.

Usage:
    python ingest.py

Reads all *.md files from docs/ directory, splits by ## headings into chunks,
embeds each chunk, and stores in rag.documents.

Env vars (from .env):
    GEMINI_API_KEY or (EMBEDDING_API_BASE + EMBEDDING_API_KEY)
    EMBEDDING_MODEL    default: gemini-embedding-001
    DATABASE_URL       default: postgresql://langfuse:langfuse@localhost:5432/langfuse
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import psycopg2

sys.path.insert(0, os.path.dirname(__file__))
from agent.tools.embed import embed

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://langfuse:langfuse@localhost:5432/langfuse",
)
DOCS_DIR = Path(__file__).parent / "docs"
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "100"))
MAX_CHUNK_CHARS = 1000


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
                source      TEXT DEFAULT 'local-docs',
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


def split_markdown(text: str, source: str) -> list[dict]:
    """Split markdown by ## headings into chunks. Further split large chunks by paragraph."""
    chunks = []
    current_heading = ""
    current_lines = []

    for line in text.splitlines():
        if line.startswith("## "):
            # Save previous chunk
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.extend(_split_large_chunk(chunk_text, source))
            current_heading = line
            current_lines = [line]
        elif line.startswith("# "):
            # Top-level heading — skip as standalone chunk, used as context only
            current_heading = line
            current_lines = [line]
        else:
            current_lines.append(line)

    # Last chunk
    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.extend(_split_large_chunk(chunk_text, source))

    return chunks


def _split_large_chunk(text: str, source: str) -> list[dict]:
    """If chunk is too large, split further by paragraph."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [{"content": text, "source": source}]

    results = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > MAX_CHUNK_CHARS and current:
            results.append({"content": current.strip(), "source": source})
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        results.append({"content": current.strip(), "source": source})
    return results


def load_docs() -> list[dict]:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"docs/ directory not found at {DOCS_DIR}")

    all_chunks = []
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {DOCS_DIR}")

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        source = f"docs/{md_file.name}"
        chunks = split_markdown(text, source)
        print(f"[ingest] {source}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"[ingest] Total: {len(all_chunks)} chunks from {len(md_files)} files.")
    return all_chunks


def insert_batch(conn, chunks: list[dict], vectors: list[list[float]]):
    with conn.cursor() as cur:
        for chunk, vec in zip(chunks, vectors):
            cur.execute(
                """
                INSERT INTO rag.documents (content, embedding, content_tsv, source)
                VALUES (%s, %s, to_tsvector('simple', %s), %s)
                """,
                (chunk["content"], vec, chunk["content"], chunk["source"]),
            )
    conn.commit()


def main():
    conn = psycopg2.connect(DATABASE_URL)
    init_schema(conn)
    truncate(conn)

    chunks = load_docs()
    total = len(chunks)
    inserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_texts = [c["content"] for c in batch]
        print(f"[ingest] Embedding batch {i // BATCH_SIZE + 1} / {(total + BATCH_SIZE - 1) // BATCH_SIZE} ...")
        vectors = embed(batch_texts)
        insert_batch(conn, batch, vectors)
        inserted += len(batch)
        print(f"[ingest] {inserted}/{total} chunks inserted.")

    conn.close()
    print(f"[ingest] Done. {inserted} chunks ingested from docs/.")


if __name__ == "__main__":
    main()

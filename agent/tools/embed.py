"""Dual-mode embedding: Gemini or OpenAI-compatible endpoint."""
import os
from typing import Union

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns a list of embedding vectors."""
    if EMBEDDING_API_BASE and EMBEDDING_API_KEY:
        return _embed_openai_compat(texts)
    return _embed_gemini(texts)


def embed_one(text: str) -> list[float]:
    """Embed a single text and return its vector."""
    return embed([text])[0]


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return [e.values for e in result.embeddings]


def _embed_openai_compat(texts: list[str]) -> list[list[float]]:
    import litellm
    response = litellm.embedding(
        model=f"openai/{EMBEDDING_MODEL}",
        input=texts,
        api_base=EMBEDDING_API_BASE,
        api_key=EMBEDDING_API_KEY,
    )
    return [item["embedding"] for item in response["data"]]

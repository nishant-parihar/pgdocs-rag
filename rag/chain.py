import os
from groq import Groq
from rag.retriever import retrieve
from typing import Optional

RELEVANCE_THRESHOLD = 0.3
HISTORY_TURNS = 6           # last N messages (each turn = 1 user + 1 assistant)
MODEL = "llama-3.1-8b-instant"

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {chunk['source_name']}\n{chunk['text']}")
    return "\n\n".join(parts)


def answer(query: str, chat_history: Optional[list] = None) -> dict:
    """
    Run the full RAG chain.

    chat_history: list of {"role": "user"|"assistant", "content": "..."} dicts.
    Returns: {"answer": str, "sources": [{"name": str, "url": str}], "rejected": bool}
    """
    chat_history = chat_history or []
    chunks = retrieve(query)

    # ── Irrelevant query guard ─────────────────────────────────────────────────
    if not chunks or chunks[0]["score"] < RELEVANCE_THRESHOLD:
        return {
            "answer": "This question doesn't appear to be related to PostgreSQL documentation.",
            "sources": [],
            "rejected": True,
        }

    # ── Build prompt ───────────────────────────────────────────────────────────
    context_block = _build_context_block(chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided PostgreSQL documentation excerpts. "
        "Always cite the source name(s) you used, e.g. (Source: Sql Select). "
        "If the context does not contain enough information to answer, say so."
    )

    # Trim history to last HISTORY_TURNS messages
    trimmed_history = chat_history[-HISTORY_TURNS:]

    messages = (
        [{"role": "system", "content": system_prompt}]
        + [{"role": "user", "content": f"Context:\n{context_block}"}]
        + [{"role": "assistant", "content": "Understood. I will answer using only the above context."}]
        + trimmed_history
        + [{"role": "user", "content": query}]
    )

    # ── LLM call ───────────────────────────────────────────────────────────────
    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,        # low temp = factual, less creative
    )

    answer_text = response.choices[0].message.content.strip()

    # Deduplicate sources preserving order
    seen = set()
    sources = []
    for chunk in chunks:
        if chunk["source_name"] not in seen:
            seen.add(chunk["source_name"])
            sources.append({"name": chunk["source_name"], "url": chunk["source_url"]})

    return {"answer": answer_text, "sources": sources, "rejected": False}
import os
import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "postgres_docs"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

# Loaded once at import time — shared across chain.py and eval.py
_model = SentenceTransformer("all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)
_collection = _client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed query and return top-k matching chunks.
    Each result dict has: text, source_url, source_name, score.
    Score is in (0, 1] — higher means more similar.
    """
    query_embedding = _model.encode(query).tolist()

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source_url": meta["source_url"],
            "source_name": meta["source_name"],
            "score": round(1 / (1 + dist), 4),
        })

    return chunks
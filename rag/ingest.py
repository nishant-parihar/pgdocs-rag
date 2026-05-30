import os
import re
import json
import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "postgres_docs"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
URLS_FILE = os.path.join(BASE_DIR, "data", "urls.json")

with open(URLS_FILE) as f:
    URLS = json.load(f)


def url_to_name(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1].replace(".html", "")
    return slug.replace("-", " ").title()


def fetch_and_parse(url: str) -> str:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup.find_all(["nav", "header", "footer", "aside"]):
        tag.decompose()

    main = (
    soup.find("div", {"class": "sect1"})
    or soup.find("div", {"class": "chapter"})
    or soup.find("div", id=re.compile(r"^(SQL|sql)"))
    or soup.find("body")
    or soup
    )
    if main is None:
        main = soup.find("body") or soup

    text = main.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    doc_ids, embeddings_list, documents, metadatas = [], [], [], []

    for url in URLS:
        name = url_to_name(url)
        print(f"Fetching: {name} ...")
        try:
            text = fetch_and_parse(url)
        except Exception as e:
            print(f"  SKIP ({e})")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 50:
                continue
            doc_id = f"{name}__chunk{i}"
            embedding = model.encode(chunk).tolist()
            doc_ids.append(doc_id)
            embeddings_list.append(embedding)
            documents.append(chunk)
            metadatas.append({"source_url": url, "source_name": name})

    collection.add(
        ids=doc_ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"\nDone. Stored {len(doc_ids)} chunks across {len(URLS)} docs.")


if __name__ == "__main__":
    ingest()
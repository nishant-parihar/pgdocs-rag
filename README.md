# PostgreSQL 16 Docs RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about PostgreSQL 16 using the official documentation as its knowledge base.

## Stack

| Component | Tool | Reason |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, no API key, fast, good semantic quality for English technical text |
| Vector store | ChromaDB (on-disk) | Persistent, lightweight, no server setup needed |
| LLM | Groq API — `llama-3.1-8b-instant` | Free tier, ~200 tokens/sec, sufficient for Q&A over provided context |
| UI | Streamlit | Zero-boilerplate chat interface |

## Project Structure
    postgres-rag/
    ├── data/
    │   └── urls.json
    ├── embeddings/
    ├── rag/
    │   ├── ingest.py
    │   ├── retriever.py
    │   └── chain.py
    ├── app/
    │   └── streamlit_app.py
    ├── eval/
    │   ├── qa_dataset.json
    │   └── eval.py
    ├── requirements.txt
    └── README.md

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd postgres-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your Groq API key

Create a `.env` file at the project root:
GROQ_API_KEY=your_groq_api_key_here

Get a free key at https://console.groq.com

### 3. Run ingestion (one-time)

```bash
python -m rag.ingest
```

This fetches all 35 PostgreSQL 16 doc pages, chunks them, embeds them locally, and stores them in `embeddings/`. Takes 2-3 minutes. Never needs to run again unless you change `data/urls.json`.

### 4. Start the app

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501

### 5. Run evaluation

```bash
python eval/eval.py
```

Prints per-question Recall@5 and cosine similarity, plus summary averages.

## How It Works

### Data Ingestion

- Fetches 35 PostgreSQL 16 documentation pages
- Strips navigation boilerplate using BeautifulSoup, targeting `sect1` and `chapter` div classes
- Splits content into ~500 character chunks with 50 character overlap
- Embeds each chunk using `all-MiniLM-L6-v2` running fully locally
- Stores chunks, embeddings, and metadata (source URL, source name) in ChromaDB on disk

### Retrieval

- Embeds the user query with the same model used during ingestion
- Queries ChromaDB for the top-5 most similar chunks using L2 distance
- Converts distance to a similarity score: `1 / (1 + distance)` → range (0, 1]

### LLM Chain

- If the top chunk score is below 0.3, rejects the query before calling the LLM
- Otherwise builds a prompt with: system instruction → retrieved chunks → chat history (last 6 messages) → current query
- Calls Groq API with `llama-3.1-8b-instant` at temperature 0.2

### Evaluation

- **Recall@5** — checks if the expected source document appears in the top-5 retrieved chunks
- **Cosine similarity** — embeds both the generated answer and reference answer, computes cosine similarity between them as a proxy for answer quality

## Results

| Metric | Score |
|---|---|
| Recall@5 | 1.00 (20/20) |
| Avg Cosine Similarity | 0.66 |

## Known Limitations

- **No re-ranking** — retrieval is pure vector similarity with no cross-encoder re-ranking step
- **Fixed chunk size** — 500 character chunks work well for most pages but may split important context across chunk boundaries
- **Similarity threshold is a rough guard** — the 0.3 threshold prevents most irrelevant queries from reaching the LLM but is not foolproof; borderline queries may slip through
- **No re-ingestion on startup** — if PostgreSQL docs are updated, ingestion must be re-run manually
- **Cold start latency** — first load takes 30-60 seconds as ChromaDB and sentence-transformers models load from disk
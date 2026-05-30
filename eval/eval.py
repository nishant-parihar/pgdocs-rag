import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.retriever import retrieve
from rag.chain import answer

DATASET_PATH = os.path.join(os.path.dirname(__file__), "qa_dataset.json")
K = 5

with open(DATASET_PATH) as f:
    dataset = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate():
    recall_hits = 0
    similarity_scores = []

    print(f"{'#':<4} {'Question':<52} {'Recall':>8} {'CosSim':>8}")
    print("-" * 76)

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        reference = item["reference_answer"]
        expected_source = item["expected_source"]

        # ── Recall@k ──────────────────────────────────────────────────────────
        chunks = retrieve(question, k=K)
        retrieved_sources = [c["source_name"] for c in chunks]
        hit = 1 if expected_source in retrieved_sources else 0
        recall_hits += hit

        # ── Answer similarity ──────────────────────────────────────────────────
        result = answer(question)
        generated = result["answer"]

        ref_embedding = model.encode(reference).tolist()
        gen_embedding = model.encode(generated).tolist()
        sim = cosine_similarity(ref_embedding, gen_embedding)
        similarity_scores.append(sim)

        print(f"{i:<4} {question[:50]:<52} {'HIT' if hit else 'MISS':>8} {sim:>8.4f}")

    print("-" * 76)
    recall_at_k = recall_hits / len(dataset)
    avg_similarity = float(np.mean(similarity_scores))
    print(f"\nRecall@{K}:          {recall_at_k:.4f}  ({recall_hits}/{len(dataset)} hits)")
    print(f"Avg Cosine Sim:     {avg_similarity:.4f}")


if __name__ == "__main__":
    evaluate()
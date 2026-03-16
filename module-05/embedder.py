# --- LESSON NOTES ---
# Module 05, Lesson 2 & 3: Embeddings — turning text into numbers
#
# An embedding is a list of ~1,000 floating-point numbers that represents
# the *meaning* of a piece of text. Two pieces of text with similar meanings
# will have similar embeddings (close together in vector space).
#
# We use Anthropic's `voyage-3` embedding model via the `voyageai` SDK.
# (Voyage AI is Anthropic's partner for embeddings.)
# ---------------------

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# voyageai SDK: pip install voyageai
try:
    import voyageai
    _voyage_client = voyageai.Client()   # reads VOYAGE_API_KEY from env
    EMBEDDING_BACKEND = "voyage"
except ImportError:
    _voyage_client = None
    EMBEDDING_BACKEND = "fallback"


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of strings into embedding vectors.

    Returns a list of float lists, one per input text.
    Falls back to a simple TF-IDF-style vector if voyageai is not installed.
    """
    if EMBEDDING_BACKEND == "voyage" and _voyage_client:
        result = _voyage_client.embed(texts, model="voyage-3", input_type="document")
        return result.embeddings

    # ── Fallback: simple bag-of-words vector (for demo without voyage key) ──
    print("⚠️  voyageai not available. Using bag-of-words fallback embeddings.")
    return _bow_embed(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    if EMBEDDING_BACKEND == "voyage" and _voyage_client:
        result = _voyage_client.embed([query], model="voyage-3", input_type="query")
        return result.embeddings[0]
    return _bow_embed([query])[0]


# ── Fallback: bag-of-words embeddings (for classroom use without API key) ───
import math
import re
from collections import Counter

_VOCAB: dict = {}

def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())

def _bow_embed(texts: List[str]) -> List[List[float]]:
    """Very simple bag-of-words embedding — good enough to demo RAG logic."""
    global _VOCAB
    all_tokens = [_tokenise(t) for t in texts]

    # Build vocabulary from this batch
    for tokens in all_tokens:
        for tok in tokens:
            if tok not in _VOCAB:
                _VOCAB[tok] = len(_VOCAB)

    dim = len(_VOCAB)
    vectors = []
    for tokens in all_tokens:
        vec = [0.0] * dim
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        for tok, cnt in counts.items():
            if tok in _VOCAB:
                vec[_VOCAB[tok]] = cnt / total
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses multi-layer neural networks.",
        "Paris is the capital city of France.",
    ]
    embeddings = embed_texts(texts)
    print(f"Backend       : {EMBEDDING_BACKEND}")
    print(f"Texts embedded: {len(embeddings)}")
    print(f"Vector length : {len(embeddings[0])}")

    # Cosine similarity between first two (should be high) and first+third (low)
    def cosine(a, b):
        dot  = sum(x * y for x, y in zip(a, b))
        na   = math.sqrt(sum(x * x for x in a))
        nb   = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    import math
    print(f"\nSimilarity (ML text vs Deep Learning text) : {cosine(embeddings[0], embeddings[1]):.3f}  ← should be high")
    print(f"Similarity (ML text vs Paris/France text)  : {cosine(embeddings[0], embeddings[2]):.3f}  ← should be low")

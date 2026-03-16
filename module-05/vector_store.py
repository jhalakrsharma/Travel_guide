# --- LESSON NOTES ---
# Module 05, Lesson 3: Vector stores — storing and searching chunks
#
# A vector store saves embeddings and lets you query them by similarity.
# We use ChromaDB — it runs fully locally (no server needed) and stores
# data in a folder on disk.
#
# Core operations:
#   add()    → store chunk text + its embedding
#   query()  → find the top-k most similar chunks for a question
#   reset()  → wipe the collection (useful when re-indexing)
# ---------------------

import os
from typing import List, Tuple
from pathlib import Path
import chromadb


CHROMA_DIR = ".chromadb"


def _get_collection(collection_name: str = "studybuddy"):
    """Return (or create) a ChromaDB collection stored on disk."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # use cosine similarity
    )


def add_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    collection_name: str = "studybuddy",
) -> None:
    """
    Store text chunks and their embeddings in the vector store.
    Each chunk gets a simple id: chunk_0, chunk_1, …
    """
    collection = _get_collection(collection_name)

    collection.add(
        ids        = [f"chunk_{i}" for i in range(len(chunks))],
        documents  = chunks,
        embeddings = embeddings,
    )
    print(f"✅ Stored {len(chunks)} chunks in collection '{collection_name}'")


def query_chunks(
    query_embedding: List[float],
    top_k: int = 3,
    collection_name: str = "studybuddy",
) -> List[Tuple[str, float]]:
    """
    Find the top-k most similar chunks for a query embedding.

    Returns a list of (chunk_text, similarity_score) tuples,
    sorted by similarity descending.
    """
    collection = _get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    chunks     = results["documents"][0]
    distances  = results["distances"][0]
    # ChromaDB cosine distance = 1 - similarity → convert back
    scores     = [1.0 - d for d in distances]

    return list(zip(chunks, scores))


def reset_collection(collection_name: str = "studybuddy") -> None:
    """Delete and recreate the collection (use when re-indexing a new doc)."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"🗑  Collection '{collection_name}' reset")


def chunk_count(collection_name: str = "studybuddy") -> int:
    return _get_collection(collection_name).count()


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from embedder import embed_texts, embed_query

    sample_chunks = [
        "Gradient descent minimises a loss function by iteratively updating parameters.",
        "Overfitting occurs when a model memorises training data and fails on new data.",
        "Paris is the capital of France.",
    ]

    reset_collection("test_col")
    embeddings = embed_texts(sample_chunks)
    add_chunks(sample_chunks, embeddings, "test_col")

    print(f"\nChunks in store: {chunk_count('test_col')}")

    q_emb = embed_query("How does gradient descent work?")
    results = query_chunks(q_emb, top_k=2, collection_name="test_col")

    print("\nTop results for 'How does gradient descent work?':")
    for text, score in results:
        print(f"  score={score:.3f}  →  {text[:80]}")

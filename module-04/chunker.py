# --- LESSON NOTES ---
# Module 04, Lesson 3: Chunking a document
#
# Why chunk?
# If a document is longer than the context window we can't send it all at
# once. We split it into overlapping chunks so nothing is cut off mid-idea.
#
# Strategy used here: fixed-size token chunks with overlap.
#   chunk_size   — max tokens per chunk
#   overlap      — tokens shared between adjacent chunks (preserves context
#                  across chunk boundaries)
#
# In Module 05 we'll use smarter semantic chunking for RAG.
# ---------------------

import tiktoken
from typing import List


ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def chunk_document(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 150,
) -> List[str]:
    """
    Split a document into token-sized chunks with overlap.

    Returns a list of text strings, each ≤ chunk_size tokens.
    Adjacent chunks share `overlap` tokens so context is preserved
    across boundaries.
    """
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(ENCODING.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap   # step forward, keeping overlap

    return chunks


def select_best_chunk(chunks: List[str], question: str) -> str:
    """
    Naive chunk selection: return the chunk whose token overlap with the
    question is highest. (Module 05 replaces this with proper vector search.)
    """
    question_words = set(question.lower().split())

    def score(chunk: str) -> int:
        chunk_words = set(chunk.lower().split())
        return len(question_words & chunk_words)

    return max(chunks, key=score)


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = " ".join(["word"] * 5000)   # 5,000 token fake document
    chunks = chunk_document(sample, chunk_size=1500, overlap=150)
    print(f"Document tokens : {count_tokens(sample):,}")
    print(f"Chunks produced : {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {count_tokens(c):,} tokens")

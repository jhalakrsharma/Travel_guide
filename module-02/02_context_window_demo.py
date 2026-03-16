# --- LESSON NOTES ---
# Module 01, Lesson 2: Context windows
#
# Key idea: Every LLM has a maximum number of tokens it can "see" at once —
# the context window. This includes EVERYTHING: your system prompt, the whole
# conversation history, and the document you're analysing.
#
# For our Study Buddy, this matters a lot:
#   - A short document fits in context easily → simple approach works
#   - A long book does NOT fit → we need RAG (covered in Module 05)
#
# This demo simulates what happens when you overflow a context window.
# ---------------------

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# Claude's context windows (approximate, for illustration)
CONTEXT_LIMITS = {
    "claude-haiku-4-5":    200_000,
    "claude-sonnet-4":     200_000,
    "claude-opus-4":       200_000,
}

SYSTEM_PROMPT = """You are a helpful study assistant. 
Answer questions based only on the document provided."""

SAMPLE_QUESTION = "What is the main topic of this document?"


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def check_fits_in_context(document: str, model: str = "claude-sonnet-4") -> None:
    """Check whether a document fits in the context window for a given model."""
    limit = CONTEXT_LIMITS[model]

    system_tokens  = count_tokens(SYSTEM_PROMPT)
    doc_tokens     = count_tokens(document)
    question_tokens = count_tokens(SAMPLE_QUESTION)
    # Reserve space for the model's reply
    reply_buffer   = 1_000

    total = system_tokens + doc_tokens + question_tokens + reply_buffer
    fits  = total < limit

    print(f"\n── Context window check for {model} ──")
    print(f"  System prompt : {system_tokens:>7,} tokens")
    print(f"  Document      : {doc_tokens:>7,} tokens")
    print(f"  Question      : {question_tokens:>7,} tokens")
    print(f"  Reply buffer  : {reply_buffer:>7,} tokens")
    print(f"  ─────────────────────────────")
    print(f"  Total         : {total:>7,} tokens")
    print(f"  Model limit   : {limit:>7,} tokens")
    print(f"  Result        : {'✅ FITS' if fits else '❌ TOO LONG — need RAG!'}")


# --- Demo 1: A short document (fits easily) ---
short_doc = """
Machine learning is a method of data analysis that automates analytical
model building. It is based on the idea that systems can learn from data,
identify patterns, and make decisions with minimal human intervention.
"""
check_fits_in_context(short_doc)


# --- Demo 2: Simulate a very long document ---
# We generate a fake long document by repeating paragraphs
paragraph = (
    "Neural networks are computing systems inspired by biological neural "
    "networks. They consist of layers of interconnected nodes that process "
    "information using connectionist approaches to computation. " * 10
)
long_doc = paragraph * 200   # ~200,000 words worth

check_fits_in_context(long_doc)

print("\n💡 Lesson takeaway:")
print("   When a document is too long to fit, we need a smarter strategy.")
print("   That strategy is called RAG — Retrieval Augmented Generation.")
print("   We'll build it in Module 05.")

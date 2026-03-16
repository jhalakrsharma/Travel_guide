# --- LESSON NOTES ---
# Module 01, Lesson 1: How LLMs see text
#
# Key idea: LLMs don't read characters or words — they read "tokens".
# A token is roughly 3-4 characters. Common words are 1 token; rare
# words or names may be split into several. This matters because:
#   - API costs are based on token count
#   - Context limits are measured in tokens
#   - Strange model behaviour often traces back to tokenisation
#
# This demo uses the `tiktoken` library (OpenAI's tokeniser).
# Claude uses a similar but proprietary tokeniser; the counts are close.
# ---------------------

import tiktoken

# We use the cl100k_base encoding — similar to what Claude uses internally
enc = tiktoken.get_encoding("cl100k_base")


def tokenise(text: str) -> None:
    """Show how a piece of text is broken into tokens."""
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]

    print(f"\nInput:  {text!r}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {len(tokens)} tokens")
    print("Breakdown:")
    for i, (tok_id, tok_str) in enumerate(zip(tokens, decoded)):
        print(f"  [{i}] id={tok_id:6d}  text={tok_str!r}")


# --- Demo 1: Plain English ---
tokenise("Hello, world!")

# --- Demo 2: A sentence (notice spaces are part of tokens) ---
tokenise("The quick brown fox jumps over the lazy dog.")

# --- Demo 3: A technical term ---
tokenise("transformer")

# --- Demo 4: An unusual word — gets split into sub-tokens ---
tokenise("supercalifragilisticexpialidocious")

# --- Demo 5: Numbers are often split per digit ---
tokenise("The year is 2025.")

# --- Demo 6: Cost estimation helper ---
def estimate_cost(text: str, cost_per_million_tokens: float = 3.0) -> None:
    n = len(enc.encode(text))
    cost = (n / 1_000_000) * cost_per_million_tokens
    print(f"\n'{text[:40]}...' → {n} tokens → ${cost:.6f} (at ${cost_per_million_tokens}/M tokens)")

sample_doc = """
Machine learning is a subset of artificial intelligence that gives computers
the ability to learn without being explicitly programmed. It focuses on developing
algorithms that can access data and use it to learn for themselves.
"""
estimate_cost(sample_doc)

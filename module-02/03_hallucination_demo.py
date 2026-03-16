# --- LESSON NOTES ---
# Module 01, Lesson 3: Hallucinations — why models confabulate
#
# Key idea: LLMs predict the most statistically likely next token.
# They don't "look things up" — they generate. When asked about
# something outside their training data, they may produce fluent,
# confident-sounding text that is simply wrong.
#
# This file does NOT call the API. Instead it illustrates the
# *mechanism* behind hallucination using a simple probability analogy,
# and shows the mitigation we'll use throughout this course:
# "grounding" the model in a real document.
# ---------------------

import random

# ── Analogy: text completion as probability ──────────────────────────────
# Imagine the model has seen millions of sentences. Given a prefix,
# it picks the next word based on learned probabilities.

completions = {
    "The capital of France is": [
        ("Paris",    0.97),
        ("Lyon",     0.02),
        ("Brussels", 0.01),
    ],
    "The inventor of Python was": [
        ("Guido van Rossum",  0.92),
        ("Linus Torvalds",    0.05),
        ("James Gosling",     0.03),
    ],
    # For obscure or fictional facts, probabilities spread out →
    # the model may pick a plausible-sounding but wrong answer
    "The first GenAI course was created by": [
        ("Andrew Ng",           0.30),
        ("Andrej Karpathy",     0.25),
        ("Geoffrey Hinton",     0.20),
        ("some unknown person", 0.15),
        ("a research lab",      0.10),
    ],
}


def simulate_completion(prefix: str, n_samples: int = 5) -> None:
    options  = completions[prefix]
    words    = [w for w, _ in options]
    weights  = [p for _, p in options]

    print(f"\nPrompt: '{prefix} ___'")
    print("Top completions the model has learned:")
    for word, prob in options:
        bar = "█" * int(prob * 30)
        print(f"  {prob:.0%}  {bar:<30}  {word!r}")

    print(f"\nSampling {n_samples} times:")
    for _ in range(n_samples):
        pick = random.choices(words, weights=weights, k=1)[0]
        print(f"  → '{prefix} {pick}'")


random.seed(42)
simulate_completion("The capital of France is")
simulate_completion("The inventor of Python was")
simulate_completion("The first GenAI course was created by")


# ── The mitigation: grounding ────────────────────────────────────────────
print("\n" + "="*60)
print("MITIGATION: Grounding the model in a real document")
print("="*60)

GROUNDED_SYSTEM_PROMPT = """You are a study assistant.
Answer questions ONLY using the information in the document below.
If the answer is not in the document, say "I don't know based on the provided text."

DOCUMENT:
{document}
"""

sample_document = """
Python was created by Guido van Rossum and first released in 1991.
It emphasises code readability and uses significant indentation.
"""

question = "Who created Python and when was it first released?"

filled_prompt = GROUNDED_SYSTEM_PROMPT.format(document=sample_document)

print("\nSystem prompt (with grounding):")
print(filled_prompt)
print(f"User question: {question}")
print("\n→ Because the answer IS in the document, the model will use it.")
print("→ Because we said 'ONLY use the document', fabrication is discouraged.")
print("\n💡 This grounding pattern is the foundation of our Study Buddy.")

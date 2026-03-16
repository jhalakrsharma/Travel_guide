# --- LESSON NOTES ---
# Module 03, Lesson 3: Chain-of-thought prompting
#
# Asking the model to "think step by step" before giving its answer
# dramatically improves accuracy on reasoning tasks. This is called
# chain-of-thought (CoT) prompting.
#
# For our Study Buddy, CoT is useful when students ask questions that
# require the model to synthesise information from multiple parts of a doc.
# ---------------------

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

PASSAGE = """
The human immune system has two main branches: the innate immune system,
which provides a rapid, non-specific response to pathogens, and the
adaptive immune system, which mounts a slower but highly specific response
and retains memory of previous infections.

T-cells are central to adaptive immunity. Helper T-cells coordinate the
immune response, while cytotoxic T-cells directly kill infected cells.
B-cells produce antibodies that neutralise pathogens.

Vaccines work by introducing a harmless form of a pathogen (or its antigens)
to train the adaptive immune system, so it can respond quickly to future
real infections.
"""

QUESTION = "Based on the passage, explain step by step how a vaccine protects you from a real infection."

# ── Without CoT ───────────────────────────────────────────────────────────
print("=== Without chain-of-thought ===")
r = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=300,
    system=f"Answer only from this passage:\n\n{PASSAGE}",
    messages=[{"role": "user", "content": QUESTION}]
)
print(r.content[0].text, "\n")

# ── With CoT ──────────────────────────────────────────────────────────────
print("=== With chain-of-thought ===")
r = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=400,
    system=f"Answer only from this passage:\n\n{PASSAGE}",
    messages=[{
        "role": "user",
        "content": (
            f"{QUESTION}\n\n"
            "Think through this step by step before giving your final answer."
        )
    }]
)
print(r.content[0].text)

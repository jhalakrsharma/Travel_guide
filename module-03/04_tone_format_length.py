# --- LESSON NOTES ---
# Module 03, Lesson 4: Controlling tone, format, and length
#
# The same question, answered four different ways — just by changing
# the system prompt instructions. This is a core skill: your prompt
# is the only control panel you have over the output shape.
#
# Formats we'll use in the Study Buddy:
#   - Bullet points for summaries
#   - Plain prose for explanations
#   - JSON for structured data extraction (Module 05)
# ---------------------

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

PASSAGE = """
Photosynthesis is the process by which green plants convert sunlight into food.
Using chlorophyll in their leaves, plants absorb sunlight and carbon dioxide
from the air, and water from the soil. Through a series of chemical reactions,
these are converted into glucose (a sugar) and oxygen. The glucose is used as
energy by the plant, and the oxygen is released into the atmosphere.
"""

QUESTION = "Explain how photosynthesis works."


def ask(label: str, system: str) -> None:
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=f"{system}\n\nDocument:\n{PASSAGE}",
        messages=[{"role": "user", "content": QUESTION}]
    )
    print(f"\n=== {label} ===")
    print(r.content[0].text)


# ── Four output styles ────────────────────────────────────────────────────
ask(
    "Plain explanation (default)",
    "You are a study assistant. Answer only from the document."
)

ask(
    "Bullet-point summary",
    "You are a study assistant. Answer only from the document. "
    "Always respond with a bullet-point list. Max 5 bullets."
)

ask(
    "ELI5 (Explain like I'm 5)",
    "You are a study assistant. Answer only from the document. "
    "Explain as if speaking to a 5-year-old. Use simple words and a fun analogy."
)

ask(
    "Structured JSON",
    "You are a study assistant. Answer only from the document. "
    "Respond ONLY with a valid JSON object in this shape: "
    '{"summary": "...", "inputs": ["..."], "outputs": ["..."], "key_molecule": "..."}'
    " No other text, no markdown fences."
)


# --- LESSON NOTES ---
# Module 03, Lesson 1: System prompts vs user prompts
#
# system  → the model's "job description". Set once per session.
#            Use it for: persona, rules, output format, grounding docs.
# user    → the human turn. Changes every message.
# assistant → the model's reply (you can pre-fill it to steer the tone).
# ---------------------

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

QUESTION = "Can you explain what a transformer model is?"

# ── Without a system prompt ───────────────────────────────────────────────
print("=== No system prompt ===")
r = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    messages=[{"role": "user", "content": QUESTION}]
)
print(r.content[0].text, "\n")

# ── With a system prompt ──────────────────────────────────────────────────
print("=== With system prompt (teacher persona) ===")
r = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    system=(
        "You are a university professor teaching an introductory AI course. "
        "Explain concepts clearly using analogies. Keep answers under 100 words."
    ),
    messages=[{"role": "user", "content": QUESTION}]
)
print(r.content[0].text, "\n")

# ── Pre-filling the assistant turn ───────────────────────────────────────
# Pre-filling forces the model to continue from your text — useful for
# structured outputs (e.g. start with "## Summary").
print("=== Pre-filled assistant turn ===")
r = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    messages=[
        {"role": "user",      "content": QUESTION},
        {"role": "assistant", "content": "## Summary\n"},  # model continues here
    ]
)
print(r.content[0].text)

# --- LESSON NOTES ---
# Module 03, Lesson 2: Zero-shot, one-shot, and few-shot prompting
#
# zero-shot  → no examples. Just ask.
# one-shot   → one example to show the desired format.
# few-shot   → multiple examples. Stronger signal for complex formats.
#
# For our Study Buddy, few-shot helps ensure answers are always
# grounded, concise, and in a consistent style.
# ---------------------

import json
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

SYSTEM = "You are a study assistant. Answer questions about machine learning."
QUESTION = "What is regularisation in machine learning?"


# ── Zero-shot ─────────────────────────────────────────────────────────────
print("=== Zero-shot ===")
r = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    system=SYSTEM,
    messages=[{"role": "user", "content": QUESTION}]
)
print(r.content[0].text, "\n")


# ── Few-shot: load examples from JSON file ────────────────────────────────
with open("prompts/few_shot_examples.json") as f:
    examples = json.load(f)

# Build the messages list: alternating user/assistant pairs, then the real question
few_shot_messages = []
for ex in examples:
    few_shot_messages.append({"role": "user",      "content": ex["user"]})
    few_shot_messages.append({"role": "assistant", "content": ex["assistant"]})

# Add the actual question at the end
few_shot_messages.append({"role": "user", "content": QUESTION})

print("=== Few-shot (examples loaded from prompts/few_shot_examples.json) ===")
r = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    system=SYSTEM,
    messages=few_shot_messages
)
print(r.content[0].text)
print("\n💡 Notice how few-shot keeps the answer length and style consistent.")

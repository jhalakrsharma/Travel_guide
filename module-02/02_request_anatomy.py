# --- LESSON NOTES ---
# Module 02, Lesson 2: Anatomy of a request
#
# Every parameter explained:
#
#   model        — which Claude model to use
#   max_tokens   — hard cap on the reply length (you pay per token either way)
#   system       — the "instruction manual" for the model (optional but important)
#   messages     — the conversation so far, as a list of role/content pairs
#   temperature  — randomness: 0 = deterministic, 1 = creative (default ~1)
#
# ---------------------

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

# ── Example 1: Minimal call ───────────────────────────────────────────────
print("=== Example 1: Minimal call ===")
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=128,
    messages=[{"role": "user", "content": "What is 2 + 2?"}]
)
print(response.content[0].text)


# ── Example 2: Adding a system prompt ─────────────────────────────────────
print("\n=== Example 2: With system prompt ===")
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=128,
    system="You are a pirate. Answer every question in pirate speak.",
    messages=[{"role": "user", "content": "What is 2 + 2?"}]
)
print(response.content[0].text)


# ── Example 3: Multi-turn conversation ────────────────────────────────────
# To simulate a conversation, include previous turns in the messages list.
print("\n=== Example 3: Multi-turn ===")
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[
        {"role": "user",      "content": "My name is Alex."},
        {"role": "assistant", "content": "Nice to meet you, Alex!"},
        {"role": "user",      "content": "What's my name?"},   # model should remember
    ]
)
print(response.content[0].text)


# ── Example 4: Controlling temperature ────────────────────────────────────
print("\n=== Example 4: Temperature = 0 (deterministic) ===")
for _ in range(3):
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=32,
        temperature=0,
        messages=[{"role": "user", "content": "Give me a random number between 1 and 10."}]
    )
    print(" →", r.content[0].text.strip())

print("\n=== Example 4b: Temperature = 1 (more varied) ===")
for _ in range(3):
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=32,
        temperature=1,
        messages=[{"role": "user", "content": "Give me a random number between 1 and 10."}]
    )
    print(" →", r.content[0].text.strip())

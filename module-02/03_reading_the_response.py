# --- LESSON NOTES ---
# Module 02, Lesson 3: Reading the response object
#
# The response object has more than just the text. Understanding
# all its fields helps with debugging and cost tracking.
# ---------------------

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Explain gradient descent in 2 sentences."}]
)

print("── Full response object ──────────────────────────────────────────")
print(f"id:            {response.id}")
print(f"model:         {response.model}")
print(f"stop_reason:   {response.stop_reason}")
#   'end_turn'    — model finished naturally
#   'max_tokens'  — hit your max_tokens cap (reply may be truncated!)
#   'stop_sequence' — model hit a custom stop string

print(f"\nusage:")
print(f"  input_tokens:  {response.usage.input_tokens}")
print(f"  output_tokens: {response.usage.output_tokens}")
print(f"  total_tokens:  {response.usage.input_tokens + response.usage.output_tokens}")

print(f"\ncontent blocks: {len(response.content)}")
for i, block in enumerate(response.content):
    print(f"  [{i}] type={block.type}  text={block.text!r}")

print("\n── The text you'll use in your app ──────────────────────────────")
text = response.content[0].text
print(text)

# ── Utility: safe text extraction ────────────────────────────────────────
def get_text(response) -> str:
    """Safely extract text from a response, even if there are multiple blocks."""
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    )

print("\n── Via helper function ──────────────────────────────────────────")
print(get_text(response))

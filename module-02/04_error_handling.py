# --- LESSON NOTES ---
# Module 02, Lesson 4: Error handling and rate limits
#
# Real apps must handle failures gracefully. The three errors you'll
# hit most often:
#   - AuthenticationError  → bad or missing API key
#   - RateLimitError       → too many requests; back off and retry
#   - APIStatusError       → server-side error (5xx); retry with backoff
#
# This file shows a production-ready retry wrapper you can copy into
# any project, including our Study Buddy.
# ---------------------

import os
import time
import random
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()


def call_with_retry(
    messages: list,
    system: str = "",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 512,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """
    Call the Anthropic API with exponential backoff on transient errors.

    Returns the text response as a string.
    Raises on permanent errors (auth, invalid request).
    """
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if system:
        kwargs["system"] = system

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text

        except anthropic.AuthenticationError as e:
            # No point retrying — the key is wrong
            raise SystemExit(f"❌ Authentication failed: {e}\n"
                             "Check your ANTHROPIC_API_KEY in .env") from e

        except anthropic.RateLimitError as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"⚠️  Rate limit hit (attempt {attempt}/{max_retries}). "
                  f"Retrying in {delay:.1f}s…")
            time.sleep(delay)

        except anthropic.APIStatusError as e:
            if attempt == max_retries or e.status_code < 500:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"⚠️  Server error {e.status_code} (attempt {attempt}/{max_retries}). "
                  f"Retrying in {delay:.1f}s…")
            time.sleep(delay)

    raise RuntimeError("Max retries exceeded")  # should not reach here


# ── Happy path ───────────────────────────────────────────────────────────
print("=== Normal call via retry wrapper ===")
result = call_with_retry(
    messages=[{"role": "user", "content": "What is backoff in API design?"}],
    system="Answer in one short paragraph.",
)
print(result)


# ── Simulating a bad API key ──────────────────────────────────────────────
print("\n=== What a bad API key looks like ===")
bad_client = anthropic.Anthropic(api_key="sk-ant-INVALID")
try:
    bad_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16,
        messages=[{"role": "user", "content": "hello"}]
    )
except anthropic.AuthenticationError as e:
    print(f"Caught AuthenticationError: {e.status_code} — {e.message}")

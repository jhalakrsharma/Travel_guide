# --- LESSON NOTES ---
# Module 06, Lesson 1: Persistent conversation memory
#
# LLMs are stateless — they remember nothing between API calls.
# "Memory" in a chatbot is an illusion we create by including the
# conversation history in every request.
#
# This module shows three memory strategies:
#   1. Full history         → simple but can overflow the context window
#   2. Rolling window       → keep only the last N turns (what we used so far)
#   3. Summary memory       → compress old history into a summary (advanced)
# ---------------------

import os
from typing import List, Dict
from dotenv import load_dotenv
import anthropic

load_dotenv()


class ConversationMemory:
    """
    Manages conversation history with a configurable strategy.

    strategy="window"  → keep last `max_turns` turns
    strategy="summary" → summarise old turns when window is full
    """

    def __init__(
        self,
        max_turns: int = 10,
        strategy: str = "window",   # "window" | "summary"
    ):
        self.max_turns = max_turns
        self.strategy  = strategy
        self._history: List[Dict[str, str]] = []
        self._summary: str = ""
        self._client  = anthropic.Anthropic()

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        if len(self._history) > self.max_turns * 2:
            self._compress()

    def get_messages(self) -> List[Dict[str, str]]:
        """Return the message list to pass to the API."""
        return list(self._history)

    def get_system_suffix(self) -> str:
        """
        If we have a compressed summary, return a note to prepend to
        the system prompt so the model knows about earlier context.
        """
        if self._summary:
            return f"\n\n[Earlier conversation summary: {self._summary}]"
        return ""

    def reset(self) -> None:
        self._history = []
        self._summary = ""

    def _compress(self) -> None:
        if self.strategy == "window":
            # Just drop the oldest turns
            self._history = self._history[-(self.max_turns * 2):]
            return

        if self.strategy == "summary":
            # Summarise the oldest half, keep the recent half
            half = len(self._history) // 2
            old_turns = self._history[:half]
            self._history = self._history[half:]

            conversation_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in old_turns
            )
            r = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarise this conversation in 3 bullet points, "
                        "focusing on what the user asked and what was answered:\n\n"
                        + conversation_text
                    )
                }]
            )
            new_summary = r.content[0].text
            if self._summary:
                self._summary = f"{self._summary}\n{new_summary}"
            else:
                self._summary = new_summary


# ── Quick demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mem = ConversationMemory(max_turns=4, strategy="window")
    exchanges = [
        ("user", "My name is Alex."),
        ("assistant", "Nice to meet you, Alex!"),
        ("user", "I'm studying machine learning."),
        ("assistant", "Great topic! What would you like to know?"),
        ("user", "Tell me about gradient descent."),
        ("assistant", "Gradient descent minimises a loss function…"),
        ("user", "What was the first thing I told you?"),   # should remember "Alex"
    ]
    for role, content in exchanges:
        mem.add(role, content)

    print(f"History length : {len(mem.get_messages())} messages")
    print(f"Messages kept  : {mem.max_turns * 2}")
    for m in mem.get_messages():
        print(f"  {m['role']:10} : {m['content'][:60]}")

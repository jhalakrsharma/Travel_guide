# --- LESSON NOTES ---
# Module 04, Lesson 4 & 5: The Q&A chain + conversation loop
#
# The QAChain class ties together:
#   - The document context (injected into the system prompt)
#   - Conversation history (so the model remembers earlier turns)
#   - The API call
#
# Key design decisions:
#   1. Document goes in the SYSTEM prompt (not the user message) →
#      harder to inject, and the model understands it's "background info"
#   2. History is a plain list of dicts — we append after each turn
#   3. We trim history if it gets too long (simple approach: keep last N turns)
# ---------------------

import os
from typing import List, Dict
from dotenv import load_dotenv
import anthropic

load_dotenv()

SYSTEM_TEMPLATE = """You are StudyBuddy, a helpful study assistant.

Answer questions ONLY using the document below.
If the answer is not in the document, say exactly:
"I couldn't find that in the provided document."

Be concise (2–4 sentences). Use bullet points when listing multiple items.

<document>
{document}
</document>"""

MAX_HISTORY_TURNS = 10   # keep last 10 user+assistant pairs


class QAChain:
    def __init__(self, document: str, model: str = "claude-sonnet-4-20250514"):
        self.model    = model
        self.system   = SYSTEM_TEMPLATE.format(document=document)
        self.history: List[Dict[str, str]] = []
        self.client   = anthropic.Anthropic()

    def ask(self, question: str) -> str:
        """Send a question and return the model's answer."""
        self.history.append({"role": "user", "content": question})

        # Trim old history to stay within context limits
        trimmed = self._trim_history()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.system,
            messages=trimmed,
        )

        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _trim_history(self) -> List[Dict[str, str]]:
        """Keep only the last MAX_HISTORY_TURNS turn-pairs."""
        max_messages = MAX_HISTORY_TURNS * 2  # each turn = 1 user + 1 assistant
        return self.history[-max_messages:]

    def reset(self) -> None:
        """Clear conversation history (start a new session)."""
        self.history = []


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    doc = """
    The water cycle, also called the hydrological cycle, describes the continuous
    movement of water on, above, and below the surface of the Earth.
    The main stages are: evaporation (water turns to vapour from oceans/lakes),
    condensation (vapour cools and forms clouds), precipitation (water falls as
    rain or snow), and collection (water gathers in oceans, lakes, and groundwater).
    """
    chain = QAChain(doc)
    print(chain.ask("What are the main stages of the water cycle?"))
    print()
    print(chain.ask("Which stage involves clouds forming?"))

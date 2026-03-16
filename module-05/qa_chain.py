# --- LESSON NOTES ---
# Module 05 — Updated QA chain using RAG
#
# What changed from Module 04:
#   - No longer injecting the whole document into the system prompt
#   - Instead: retrieve relevant chunks per question, inject only those
#   - History is preserved across questions (the retriever handles context)
# ---------------------

import os
from typing import List, Dict
from dotenv import load_dotenv
import anthropic
from retriever import Retriever

load_dotenv()

SYSTEM_TEMPLATE = """You are StudyBuddy, a helpful study assistant.

Answer questions ONLY using the document chunks provided in each user message.
If the provided chunks do not contain enough information to answer, say:
"I couldn't find that in the provided document."

Be concise (2–4 sentences). Use bullet points when listing multiple items.
Do NOT use any knowledge outside of the provided chunks."""

MAX_HISTORY_TURNS = 8


class RAGQAChain:
    def __init__(
        self,
        retriever: Retriever,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.retriever = retriever
        self.model     = model
        self.history: List[Dict[str, str]] = []
        self.client    = anthropic.Anthropic()

    def ask(self, question: str) -> str:
        # 1. Retrieve relevant context
        context = self.retriever.retrieve(question)

        if context:
            user_content = (
                f"Relevant document excerpts:\n\n{context}\n\n"
                f"Question: {question}"
            )
        else:
            user_content = (
                f"[No relevant document excerpts found for this question]\n\n"
                f"Question: {question}"
            )

        # 2. Append to history and trim
        self.history.append({"role": "user", "content": user_content})
        trimmed = self.history[-(MAX_HISTORY_TURNS * 2):]

        # 3. Call the model
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_TEMPLATE,
            messages=trimmed,
        )
        answer = response.content[0].text

        # 4. Save assistant reply (without the chunk context, just the question)
        #    We replace the user message with the clean question for history
        self.history[-1] = {"role": "user", "content": question}
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self):
        self.history = []

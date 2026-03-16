# --- LESSON NOTES ---
# Module 05, Lesson 4: Connecting retrieval to the chatbot
#
# The Retriever is the glue layer: given a question, it fetches the
# most relevant chunks from the vector store and returns them as a
# single context string ready to inject into the prompt.
# ---------------------

from typing import List, Tuple
from embedder import embed_query
from vector_store import query_chunks


class Retriever:
    def __init__(
        self,
        top_k: int = 3,
        collection_name: str = "studybuddy",
        score_threshold: float = 0.3,
    ):
        self.top_k             = top_k
        self.collection_name   = collection_name
        self.score_threshold   = score_threshold

    def retrieve(self, question: str) -> str:
        """
        Embed the question, fetch top-k chunks, filter by score threshold,
        and return them joined as a single context string.
        """
        q_emb   = embed_query(question)
        results = query_chunks(q_emb, top_k=self.top_k,
                               collection_name=self.collection_name)

        # Filter out chunks that are not similar enough
        relevant = [(text, score) for text, score in results
                    if score >= self.score_threshold]

        if not relevant:
            return ""   # signal: nothing relevant found

        # Join with clear separators so the model can tell chunks apart
        parts = []
        for i, (text, score) in enumerate(relevant, 1):
            parts.append(f"[Chunk {i} | similarity={score:.2f}]\n{text}")

        return "\n\n---\n\n".join(parts)

# --- LESSON NOTES ---
# Module 05 — Updated CLI app with RAG
#
# Usage:
#   python app.py index path/to/document.txt   ← index a new document
#   python app.py chat                          ← start chatting
# ---------------------

import sys
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Reuse the loader and chunker from Module 04
sys.path.insert(0, str(Path(__file__).parent.parent / "module-04-study-buddy-v1"))
from document_loader import load_document, get_word_count
from chunker import chunk_document

from embedder import embed_texts, embed_query
from vector_store import add_chunks, reset_collection, chunk_count
from retriever import Retriever
from qa_chain import RAGQAChain

BANNER = """
╔══════════════════════════════════════════════╗
║          📚 StudyBuddy  v2.0  (RAG)          ║
║   Semantic search over your documents        ║
║   Type  'quit'  to exit                      ║
║   Type  'reset' to clear conversation        ║
╚══════════════════════════════════════════════╝
"""


def index_document(doc_path: str) -> None:
    """Load, chunk, embed, and store a document."""
    print(f"\nIndexing: {doc_path}")
    text   = load_document(doc_path)
    chunks = chunk_document(text, chunk_size=400, overlap=50)
    print(f"  Words   : {get_word_count(text):,}")
    print(f"  Chunks  : {len(chunks)}")
    print(f"  Embedding {len(chunks)} chunks …")

    reset_collection()
    embeddings = embed_texts(chunks)
    add_chunks(chunks, embeddings)
    print(f"✅ Indexed {len(chunks)} chunks")


def chat() -> None:
    """Start the interactive chat loop."""
    if chunk_count() == 0:
        print("No documents indexed yet. Run:  python app.py index <path>")
        return

    retriever = Retriever(top_k=3, score_threshold=0.2)
    chain     = RAGQAChain(retriever)
    print(BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            chain.reset()
            print("🔄 Conversation cleared.\n")
            continue

        answer = chain.ask(user_input)
        print(f"\nStudyBuddy: {answer}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python app.py index <document.txt>")
        print("  python app.py chat")
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        if len(sys.argv) < 3:
            print("Provide a document path: python app.py index <document.txt>")
            sys.exit(1)
        index_document(sys.argv[2])

    elif command == "chat":
        chat()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

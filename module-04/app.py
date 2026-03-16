
# --- LESSON NOTES ---
# Module 04, Lesson 5: The CLI app — gluing everything together
#
# Run: python app.py sample_docs/machine_learning_intro.txt
#
# This is the first complete, working Study Buddy. It:
#   1. Loads a document from disk
#   2. Chunks it (in case it's large)
#   3. Selects the most relevant chunk per question (naive keyword matching)
#   4. Answers in a conversation loop
#
# Limitation: naive chunk selection → fixed in Module 05 with RAG.
# ---------------------

import sys
import os
from dotenv import load_dotenv

from document_loader import load_document, get_word_count
from chunker import chunk_document, select_best_chunk, count_tokens
from qa_chain import QAChain

load_dotenv()

BANNER = """
╔══════════════════════════════════════════╗
║          📚 StudyBuddy  v1.0             ║
║   Ask questions about your document     ║
║   Type  'quit'  to exit                 ║
║   Type  'reset' to clear history        ║
║   Type  'info'  to see document stats   ║
╚══════════════════════════════════════════╝
"""


def main():
    # ── Load document ──────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python app.py <path_to_document.txt>")
        sys.exit(1)

    doc_path = sys.argv[1]
    print(f"\nLoading: {doc_path} …")

    try:
        raw_text = load_document(doc_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    chunks = chunk_document(raw_text, chunk_size=1500, overlap=150)
    print(f"✅ Document loaded — {get_word_count(raw_text):,} words, "
          f"{len(chunks)} chunk(s)")

    # If short enough, use the full document; otherwise use chunk selection
    full_tokens = count_tokens(raw_text)
    use_full_doc = full_tokens <= 6000

    if use_full_doc:
        chain = QAChain(raw_text)
        print("📄 Short document — using full text as context")
    else:
        print(f"📚 Long document ({full_tokens:,} tokens) — "
              f"will select best chunk per question")

    print(BANNER)

    # ── Conversation loop ──────────────────────────────────────────────
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
            if use_full_doc:
                chain.reset()
            else:
                # re-create chain; we'll pick a new chunk on next question
                pass
            print("🔄 Conversation history cleared.\n")
            continue

        if user_input.lower() == "info":
            print(f"\n📊 Document: {doc_path}")
            print(f"   Words  : {get_word_count(raw_text):,}")
            print(f"   Tokens : {full_tokens:,}")
            print(f"   Chunks : {len(chunks)}\n")
            continue

        # ── Pick context and answer ────────────────────────────────────
        if use_full_doc:
            answer = chain.ask(user_input)
        else:
            best_chunk = select_best_chunk(chunks, user_input)
            # Fresh chain per question (no cross-chunk memory — fixed in M05)
            temp_chain = QAChain(best_chunk)
            answer = temp_chain.ask(user_input)

        print(f"\nStudyBuddy: {answer}\n")


if __name__ == "__main__":
    main()

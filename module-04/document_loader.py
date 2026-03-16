# --- LESSON NOTES ---
# Module 04, Lesson 2: Loading a document
#
# We support .txt files here. The function returns a plain Python string —
# everything downstream just works with strings, which keeps things simple.
#
# In a production app you'd also support .pdf, .docx, .md etc.
# ---------------------

import os
from pathlib import Path


def load_document(path: str) -> str:
    """
    Load a document from disk and return its content as a string.

    Supports: .txt  (and .md — treated as plain text)
    Raises FileNotFoundError if the path doesn't exist.
    Raises ValueError for unsupported file types.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8")

    # Bonus: PDF support (requires PyPDF2)
    if suffix == ".pdf":
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF support. "
                "Run: pip install PyPDF2"
            )

    raise ValueError(
        f"Unsupported file type: {suffix}. Supported types: .txt, .md, .pdf"
    )


def get_word_count(text: str) -> int:
    return len(text.split())


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_docs/machine_learning_intro.txt"
    doc = load_document(path)
    print(f"Loaded: {path}")
    print(f"Characters : {len(doc):,}")
    print(f"Words      : {get_word_count(doc):,}")
    print(f"\nFirst 300 chars:\n{doc[:300]}")

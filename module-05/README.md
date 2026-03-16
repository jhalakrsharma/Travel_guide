# Module 05 — Making It Smarter with RAG

We replace the naive "keyword chunk selection" with proper **Retrieval Augmented Generation (RAG)**:
1. Each chunk is turned into an embedding (a vector of numbers)
2. Embeddings are stored in a local vector database (ChromaDB)
3. At query time, the question is embedded and the most semantically similar chunks are retrieved
4. Those chunks are injected into the prompt — not the whole document

## Run

```bash
# Index a document (run once per document)
python app.py index sample_docs/machine_learning_intro.txt

# Ask questions
python app.py chat sample_docs/machine_learning_intro.txt

# Evaluate answer quality
python evaluate.py
```

## Files

| File | Purpose |
|------|---------|
| `embedder.py` | Convert text chunks into embedding vectors via Anthropic API |
| `vector_store.py` | Store and retrieve embeddings using ChromaDB |
| `retriever.py` | Top-k semantic chunk retrieval |
| `qa_chain.py` | Updated chain using retrieved chunks |
| `evaluate.py` | Simple answer quality evaluation |
| `app.py` | Updated CLI with RAG |

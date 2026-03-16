# Module 04 — Building the Study Buddy (v1)

The first working version of the app. By the end of this module you have a **CLI chatbot** that:
- Accepts any `.txt` document
- Chunks it to fit in the context window
- Answers questions about it in a conversation loop

## Run

```bash
python app.py sample_docs/machine_learning_intro.txt
```

Then type questions. Type `quit` to exit.

## Files

| File | Purpose |
|------|---------|
| `document_loader.py` | Read a `.txt` or `.pdf` file into a string |
| `chunker.py` | Split a document into token-safe chunks |
| `qa_chain.py` | Build the prompt and call the API |
| `app.py` | CLI entry point — glues everything together |
| `sample_docs/` | Example documents to test with |

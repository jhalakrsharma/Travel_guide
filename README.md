# 🤖 Your First GenAI Application — Course Code

> **Course use case:** An AI-powered **personal study buddy** — upload any document and ask questions about it.  
> We build it from scratch across 6 modules, adding capability in every one.

---

## 📚 Course Modules

| Module | Title | What you build |
|--------|-------|----------------|
| [01](./module-01-what-is-genai/) | What is Generative AI? | Conceptual demos — tokenisation, context windows |
| [02](./module-02-first-api-call/) | Your First API Call | A bare-bones script that talks to an LLM |
| [03](./module-03-prompt-engineering/) | Prompt Engineering Basics | Prompt templates, roles, few-shot examples |
| [04](./module-04-study-buddy-v1/) | Building the Study Buddy | Working chatbot: upload a doc, ask questions |
| [05](./module-05-rag-upgrade/) | Making It Smarter with RAG | Embeddings + vector search for large documents |
| [06](./module-06-ship-it/) | Ship It & What's Next | Memory, streaming, deployment, guardrails |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)

### Setup
```bash
git clone https://github.com/your-username/your-first-genai-app.git
cd your-first-genai-app

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Open .env and paste your ANTHROPIC_API_KEY
```

Then navigate into any module folder and follow its own `README.md`.

---

## 🗂 Repository Layout

```
your-first-genai-app/
├── README.md                  ← you are here
├── requirements.txt           ← all Python deps across all modules
├── .env.example               ← copy → .env, add your API key
│
├── module-01-what-is-genai/
│   ├── README.md
│   ├── 01_tokenisation_demo.py
│   ├── 02_context_window_demo.py
│   └── 03_hallucination_demo.py
│
├── module-02-first-api-call/
│   ├── README.md
│   ├── 01_hello_claude.py
│   ├── 02_request_anatomy.py
│   ├── 03_reading_the_response.py
│   └── 04_error_handling.py
│
├── module-03-prompt-engineering/
│   ├── README.md
│   ├── 01_system_vs_user_prompt.py
│   ├── 02_zero_one_few_shot.py
│   ├── 03_chain_of_thought.py
│   ├── 04_tone_format_length.py
│   ├── 05_prompt_injection_demo.py
│   └── prompts/
│       ├── system_studybuddy.txt
│       └── few_shot_examples.json
│
├── module-04-study-buddy-v1/
│   ├── README.md
│   ├── app.py                 ← run this: simple CLI chatbot
│   ├── document_loader.py
│   ├── chunker.py
│   ├── qa_chain.py
│   └── sample_docs/
│       └── machine_learning_intro.txt
│
├── module-05-rag-upgrade/
│   ├── README.md
│   ├── app.py                 ← upgraded app with RAG
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── qa_chain.py
│   └── evaluate.py
│
└── module-06-ship-it/
    ├── README.md
    ├── app.py                 ← final streaming web app
    ├── memory.py
    ├── guardrails.py
    ├── static/
    │   └── index.html
    └── Procfile               ← for Render.com deployment
```

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Your key from [console.anthropic.com](https://console.anthropic.com) |
| `MODEL_NAME` | Default: `claude-sonnet-4-20250514` |

---

## 📝 Notes for Instructors

- Each module is **self-contained** — students can jump to any module.
- Every script has a `# --- LESSON NOTES ---` block at the top explaining what it teaches.
- Suggested additions from the course plan (hallucinations, streaming, guardrails) are all implemented.
- Code is intentionally kept simple — no frameworks, no magic, just readable Python.

---

## 📄 License

MIT — use freely for teaching and learning.

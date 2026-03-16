# Module 06 — Ship It & What's Next

The final, production-ready version of StudyBuddy:
- **Streaming** responses (text appears word by word)
- **Persistent memory** across the session
- **Guardrails** — output filtering for safety
- **Web UI** — a minimal HTML/JS front-end
- **Deployment** — one-click deploy to Render.com

## Run locally

```bash
python app.py
# Open http://localhost:5000 in your browser
```

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo, set the build command to `pip install -r ../../requirements.txt`
4. Add `ANTHROPIC_API_KEY` as an environment variable
5. Deploy

## Files

| File | Purpose |
|------|---------|
| `app.py` | Flask web server with streaming SSE endpoint |
| `memory.py` | Persistent conversation memory |
| `guardrails.py` | Input/output safety filters |
| `static/index.html` | Minimal chat UI |
| `Procfile` | Process config for Render/Heroku |

# --- LESSON NOTES ---
# Module 06, Lesson 1 & 2: Streaming responses + Flask web server
#
# Two new ideas:
#
# 1. STREAMING — instead of waiting for the full reply, we use
#    anthropic's streaming API and send tokens to the browser as
#    Server-Sent Events (SSE). This makes the app feel alive.
#
# 2. WEB UI — a minimal Flask server serves a static HTML page
#    (static/index.html) and exposes two endpoints:
#      POST /upload   → index a new document
#      GET  /chat     → streaming SSE chat endpoint
#
# Run: python app.py
# Then open: http://localhost:5000
# ---------------------

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, Response, send_from_directory, jsonify

load_dotenv()

import anthropic

# Reuse shared modules
sys.path.insert(0, str(Path(__file__).parent.parent / "module-04-study-buddy-v1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "module-05-rag-upgrade"))

from document_loader import load_document
from chunker import chunk_document
from embedder import embed_texts
from vector_store import add_chunks, reset_collection, chunk_count
from retriever import Retriever
from memory import ConversationMemory
from guardrails import Guardrails

app = Flask(__name__, static_folder="static")

client    = anthropic.Anthropic()
retriever = Retriever(top_k=3, score_threshold=0.2)
memory    = ConversationMemory(max_turns=10, strategy="window")
guards    = Guardrails()

SYSTEM_BASE = """You are StudyBuddy, a helpful study assistant.

Answer questions ONLY using the document chunks provided with each question.
If the chunks don't contain the answer, say "I couldn't find that in the document."
Be concise (2–4 sentences). Use bullet points when listing multiple items.{memory_suffix}"""


# ── Static files ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── Upload & index a document ─────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    tmp_path = f"/tmp/{file.filename}"
    file.save(tmp_path)

    try:
        text   = load_document(tmp_path)
        chunks = chunk_document(text, chunk_size=400, overlap=50)
        reset_collection()
        embeddings = embed_texts(chunks)
        add_chunks(chunks, embeddings)
        memory.reset()
        return jsonify({"status": "ok", "chunks": len(chunks), "filename": file.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Streaming chat endpoint ───────────────────────────────────────────────
@app.route("/chat")
def chat():
    question = request.args.get("q", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Input guard
    safe, reason = guards.validate_input(question)
    if not safe:
        def blocked():
            yield f"data: {json.dumps({'text': reason, 'done': True})}\n\n"
        return Response(blocked(), mimetype="text/event-stream")

    # Retrieve context
    context = retriever.retrieve(question) if chunk_count() > 0 else ""
    if context:
        user_content = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
    else:
        user_content = question

    # Build messages with history
    memory.add("user", question)
    messages = memory.get_messages()
    # Replace last user message with the context-enriched version
    messages[-1] = {"role": "user", "content": user_content}

    system = SYSTEM_BASE.format(memory_suffix=memory.get_system_suffix())

    # Stream the response
    def generate():
        full_reply = []
        try:
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_reply.append(text)
                    yield f"data: {json.dumps({'text': text, 'done': False})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'text': f'Error: {e}', 'done': True})}\n\n"
            return

        # Output guard + save to memory
        complete = "".join(full_reply)
        safe_reply = guards.validate_output(complete)
        memory.add("assistant", safe_reply)

        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 StudyBuddy running at http://localhost:{port}")
    print("   Upload a document via the web UI, then start asking questions.\n")
    app.run(host="0.0.0.0", port=port, debug=False)

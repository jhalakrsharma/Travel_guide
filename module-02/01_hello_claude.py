# --- LESSON NOTES ---
# Module 02, Lesson 1: The simplest possible API call
#
# Three things happen here:
#   1. We load our API key from a .env file (never hardcode secrets!)
#   2. We create an Anthropic client
#   3. We send one message and print the reply
#
# Run: python 01_hello_claude.py
# ---------------------

import os
from dotenv import load_dotenv
import anthropic

# Load ANTHROPIC_API_KEY from .env
load_dotenv()

# Create the client — it reads ANTHROPIC_API_KEY from the environment automatically
client = anthropic.Anthropic()

# Send a message
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Hello! In one sentence, what is an LLM?"}
    ]
)

# The reply lives in message.content[0].text
print(message.content[0].text)

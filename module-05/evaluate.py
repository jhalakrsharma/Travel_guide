# --- LESSON NOTES ---
# Module 05, Lesson 5: Evaluating answer quality
#
# How do you know if your RAG pipeline is working?
# A simple but effective approach: create a small "golden set" of
# question/answer pairs you know are correct, then ask the LLM to
# judge each answer on a 1–5 scale.
#
# This is called "LLM-as-judge" evaluation and is widely used in industry.
# ---------------------

import os
import json
from dotenv import load_dotenv
import anthropic
from retriever import Retriever
from qa_chain import RAGQAChain

load_dotenv()
client = anthropic.Anthropic()

# ── Golden test set (question + expected key points) ──────────────────────
GOLDEN_SET = [
    {
        "question": "What is overfitting in machine learning?",
        "key_points": ["learns training data too well", "performs poorly on new data",
                        "memorises noise"],
    },
    {
        "question": "What are the three types of machine learning?",
        "key_points": ["supervised", "unsupervised", "reinforcement"],
    },
    {
        "question": "What is gradient descent used for?",
        "key_points": ["optimisation", "minimise loss", "update parameters"],
    },
    {
        "question": "What year was Python invented?",   # NOT in the document
        "key_points": ["not in document", "couldn't find"],
    },
]

JUDGE_PROMPT = """You are evaluating the quality of a study assistant's answer.

Question: {question}

Expected key points (at least some of these should appear in a good answer):
{key_points}

Assistant's answer:
{answer}

Rate the answer from 1 to 5:
  5 = Excellent: covers all key points, accurate, concise
  4 = Good: covers most key points, minor omissions
  3 = Adequate: covers some key points
  2 = Poor: misses most key points or contains inaccuracies
  1 = Wrong: completely off-track or contradicts the key points

Respond ONLY with a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def judge_answer(question: str, key_points: list, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        key_points="\n".join(f"- {p}" for p in key_points),
        answer=answer,
    )
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return json.loads(r.content[0].text)
    except json.JSONDecodeError:
        return {"score": 0, "reason": "Could not parse judge response"}


def run_evaluation():
    retriever = Retriever(top_k=3)
    chain     = RAGQAChain(retriever)

    print("Running RAG evaluation on golden set…\n")
    scores = []

    for item in GOLDEN_SET:
        chain.reset()
        answer  = chain.ask(item["question"])
        verdict = judge_answer(item["question"], item["key_points"], answer)
        scores.append(verdict["score"])

        print(f"Q: {item['question']}")
        print(f"A: {answer[:120]}{'…' if len(answer) > 120 else ''}")
        print(f"Score: {verdict['score']}/5  —  {verdict['reason']}")
        print()

    avg = sum(scores) / len(scores)
    print(f"{'─'*50}")
    print(f"Average score: {avg:.1f} / 5.0")
    print(f"{'─'*50}")
    return avg


if __name__ == "__main__":
    run_evaluation()

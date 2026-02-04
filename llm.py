# -*- coding: utf-8 -*-
"""LLM helper using Groq Python SDK only (no LangChain dependency)."""
from __future__ import annotations
import os

SYSTEM_PROMPT_EXEC = "You are a senior enterprise delivery intelligence advisor."

def explain_insight(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Calls Groq chat completions directly.
    Returns a helpful error message if the key is missing or network/SDK fails.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ LLM unavailable: GROQ_API_KEY is not set for this process."

    try:
        from groq import Groq
    except Exception as e:
        return f"❌ LLM unavailable: Groq SDK not installed/importable. ({e})"

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,  # e.g., "llama-3.1-8b-instant"
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXEC},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Surface exact server/client error so we know what's wrong
        return f"❌ LLM call failed: {e}"






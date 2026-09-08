"""Question answering over retrieved PDF chunks."""

from __future__ import annotations

import os
import re
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

load_dotenv()

llm_model = None

try:
    from langchain_groq import ChatGroq

    if os.getenv("GROQ_API_KEY"):
        llm_model = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_api_key=os.environ["GROQ_API_KEY"],
        )
except ImportError:
    pass


def get_context(documents: list[Any]) -> str:
    return "\n\n".join(document.page_content for document in documents)


def _extractive_answer(documents: list[Any], query: str) -> str:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked: list[tuple[int, str]] = []
    for document in documents:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", document.page_content.strip()):
            sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            score = len(query_terms & sentence_terms)
            if score and sentence.strip():
                ranked.append((score, sentence.strip()))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return "I could not find an answer in the uploaded PDF."
    return " ".join(sentence for _, sentence in ranked[:3])


def answer_query(documents: list[Any], model: Any, query: str) -> SimpleNamespace:
    if model is not None:
        prompt = (
            "Answer only from the following PDF context. If the answer is not present, say so.\n\n"
            f"Context:\n{get_context(documents)}\n\nQuestion: {query}"
        )
        try:
            response = model.invoke(prompt)
            return SimpleNamespace(content=getattr(response, "content", str(response)))
        except Exception:
            pass
    return SimpleNamespace(content=_extractive_answer(documents, query))


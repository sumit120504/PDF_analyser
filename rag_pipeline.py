"""Question answering over retrieved PDF chunks."""

from __future__ import annotations

import os
import re
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

llm_model = None


class OpenRouterChatModel:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "PDF Research Assistant"),
            },
        )

    def invoke(self, prompt: str) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message


openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
if openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
    llm_model = OpenRouterChatModel(
        api_key=openrouter_api_key,
        model_name=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    )


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


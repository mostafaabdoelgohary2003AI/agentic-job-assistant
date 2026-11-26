"""LLM backends.

``MockLLM`` makes the whole graph runnable and testable with no API key:
its answers are deterministic, derived from the input text with the same
extraction rules a prompted LLM would approximate. ``OpenAICompatibleLLM``
talks to any OpenAI-style chat endpoint using only the standard library.
"""

from __future__ import annotations

import json
import os
import urllib.request


class MockLLM:
    """Deterministic stand-in for a real LLM (offline demos and tests)."""

    def complete(self, system: str, user: str) -> str:
        return f"[mock-llm:{system.split()[0].lower()}] {user[:400]}"

    def complete_json(self, system: str, user: str) -> dict:
        return {}


class OpenAICompatibleLLM:
    """Minimal chat-completions client (OpenAI, NVIDIA NIM, Ollama, ...)."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.3, timeout: int = 90) -> None:
        self.model = model or os.environ.get("LLM_MODEL", "meta/llama-3.1-8b-instruct")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("an API key is required (LLM_API_KEY or OPENAI_API_KEY)")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": self.temperature,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"].strip()

    def complete_json(self, system: str, user: str) -> dict:
        raw = self.complete(system, user + "\n\nRespond with a single JSON object only.")
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"LLM did not return JSON: {raw[:200]}")
        return json.loads(raw[start : end + 1])

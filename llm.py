"""Minimal OpenRouter client (same pattern as the site-chatbot demo): Gemini 3.8 Flash with low
reasoning first, then automatic fallback. Every call returns the model used, the latency and the
real cost reported by OpenRouter, and appends one line to outputs/spend.jsonl (no prompts, no key).

The key is read only from the OPENROUTER_API_KEY environment variable. There is no default key.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "google/gemini-3.8-flash",  # main model, low reasoning
    "google/gemini-3.1-flash-lite",  # paid backup
    "nex-agi/nex-n2.5-pro:free",  # free backups (slow)
    "nvidia/nemotron-3-super-120b-a12b:free",
]
REASONING = {"google/gemini-3.8-flash": "low"}
RETRYABLE = {402, 404, 408, 429, 500, 502, 503, 504}
SPEND_LOG = Path(__file__).parent / "outputs" / "spend.jsonl"


class LLMError(Exception):
    pass


def api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or None


def _headers() -> dict:
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "X-Title": "rag-fix-case-study"}


def _reasoning(model: str) -> dict:
    effort = REASONING.get(model, "")
    return {"reasoning": {"effort": effort}} if effort else {}


def _clean_json(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    return fenced.group(1) if fenced else text


def _log(purpose: str, model: str, usage: dict, seconds: float) -> None:
    SPEND_LOG.parent.mkdir(exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": purpose, "model": model,
           "cost_usd": usage.get("cost", 0.0), "prompt_tokens": usage.get("prompt_tokens"),
           "completion_tokens": usage.get("completion_tokens"), "seconds": round(seconds, 3)}
    with SPEND_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _call(messages: list[dict], purpose: str, schema: dict | None = None, timeout: int = 90) -> dict:
    if not api_key():
        raise LLMError("Set the OPENROUTER_API_KEY environment variable.")
    errors = []
    for model in DEFAULT_MODELS:
        body = {"model": model, "messages": messages, "max_tokens": 4000, "usage": {"include": True}, **_reasoning(model)}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}}
            body["provider"] = {"require_parameters": True}
        start = time.perf_counter()
        try:
            r = requests.post(URL, headers=_headers(), json=body, timeout=timeout)
        except requests.RequestException as err:
            errors.append(f"{model}: {type(err).__name__}")
            continue
        seconds = time.perf_counter() - start
        if r.status_code == 401:
            raise LLMError("The OpenRouter API key was rejected.")
        if r.status_code in RETRYABLE or r.status_code != 200:
            errors.append(f"{model}: {r.status_code}")
            continue
        try:
            data = json.loads(r.text, strict=False)
            content = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            parsed = json.loads(_clean_json(content), strict=False) if schema else None
        except (ValueError, KeyError, IndexError, TypeError):
            errors.append(f"{model}: unusable answer")
            continue
        if not schema and not content.strip():
            errors.append(f"{model}: empty answer")
            continue
        _log(purpose, model, usage, seconds)
        return {"text": content.strip(), "json": parsed, "model": model, "seconds": seconds,
                "cost_usd": usage.get("cost", 0.0), "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}
    raise LLMError("All models failed: " + "; ".join(errors))


def complete_text(messages: list[dict], purpose: str) -> dict:
    return _call(messages, purpose)


def complete_json(messages: list[dict], schema: dict, purpose: str) -> dict:
    return _call(messages, purpose, schema=schema)

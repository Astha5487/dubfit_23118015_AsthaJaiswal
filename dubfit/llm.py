"""One swappable LLM client. Any LiteLLM model string works:
    gemini/gemini-3.8-flash, groq/openai/gpt-oss-120b, openai/gpt-4o-mini,
    anthropic/claude-sonnet-4-5, ollama/llama3.1, ...
Responses are cached on disk (same prompt -> same answer, zero cost on re-runs).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "llm"
CACHE.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "Groq GPT-OSS 120B (free tier)": ("groq/openai/gpt-oss-120b", "GROQ_API_KEY"),
    "Gemini 3.8 Flash (free tier)": ("gemini/gemini-3.8-flash", "GEMINI_API_KEY"),
    "OpenAI GPT-4o mini": ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
    "Anthropic Claude": ("anthropic/claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
    "Custom LiteLLM model": ("", ""),
}


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
    errors: list = field(default_factory=list)

    def to_dict(self):
        return self.__dict__.copy()


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


class LLM:
    def __init__(self, model: str, api_key: str | None = None, temperature: float = 0.3,
                 min_interval: float = 0.0, max_retries: int = 4, timeout: float = 150.0,
                 verbose: bool = True):
        self.model, self.api_key, self.temperature = model, api_key, temperature
        self.min_interval, self.max_retries = min_interval, max_retries
        self.timeout, self.verbose = timeout, verbose
        self.usage = Usage()
        self._last = 0.0

    def json(self, system: str, user: str) -> dict:
        key = hashlib.sha1(f"{self.model}|{self.temperature}|{system}|{user}".encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            self.usage.cached += 1
            return json.loads(path.read_text())
        if self.model.startswith("stub/"):
            out = _stub(user)
        else:
            out = self._call(system, user)
        path.write_text(json.dumps(out, ensure_ascii=False))
        return out

    def _call(self, system: str, user: str) -> dict:
        import logging

        import litellm
        litellm.suppress_debug_info = True
        litellm.drop_params = True  # silently drop options a provider does not support
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs = dict(model=self.model, messages=messages, timeout=self.timeout)
        if "gemini-3" not in self.model:  # Gemini 3 is deprecating temperature
            kwargs["temperature"] = self.temperature
        extras = {"response_format": {"type": "json_object"}}
        if self.model.startswith(("gemini/", "vertex_ai/")) or "gpt-oss" in self.model:
            # Translation needs little "thinking"; heavy thinking makes calls slow and token-hungry.
            extras["reasoning_effort"] = "low"
        if self.api_key:
            kwargs["api_key"] = self.api_key
        delay = 4.0
        rate_waits = 0
        attempt = -1
        while attempt < self.max_retries - 1:
            attempt += 1
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            if self.verbose:
                print(f"    calling {self.model} ...", end="", flush=True)
            try:
                try:
                    resp = litellm.completion(**extras, **kwargs)
                except litellm.exceptions.BadRequestError:
                    resp = litellm.completion(**kwargs)  # provider without JSON mode / reasoning option
                if self.verbose:
                    print(f" ok ({time.time() - t0:.0f}s)", flush=True)
                self._last = time.time()
                self.usage.calls += 1
                self.usage.seconds += time.time() - t0
                u = getattr(resp, "usage", None)
                if u:
                    self.usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                    self.usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0
                try:
                    self.usage.cost_usd += litellm.completion_cost(resp) or 0.0
                except Exception:
                    pass
                return _extract_json(resp.choices[0].message.content)
            except (litellm.exceptions.AuthenticationError, litellm.exceptions.NotFoundError):
                raise
            except Exception as e:  # rate limits, timeouts, bad JSON -> back off and retry
                self.usage.errors.append(f"{type(e).__name__}: {str(e)[:160]}")
                is_rate = "RateLimit" in type(e).__name__
                if self.verbose:
                    print(f" failed after {time.time() - t0:.0f}s -> {type(e).__name__}: {str(e)[:180]}", flush=True)
                if is_rate and rate_waits < 15:
                    # Per-minute token limits: wait as long as the provider asks, without using up a retry.
                    m = re.search(r"try again in ([\d.]+)s", str(e))
                    pause = min(90.0, float(m.group(1)) + 2) if m else 62.0
                    rate_waits += 1
                    if self.verbose:
                        print(f"    rate limit: waiting {pause:.0f}s", flush=True)
                    time.sleep(pause)
                    attempt -= 1
                    continue
                if attempt == self.max_retries - 1:
                    raise
                if self.verbose:
                    print(f"    retry {attempt + 2}/{self.max_retries} in {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay *= 2
        return {}


def _stub(user: str) -> dict:
    """Test-only provider: no real translation. Used by tests/ to exercise the pipeline offline."""
    data = json.loads(user)
    lines = data.get("lines", [])
    mode = data.get("task")
    out = []
    for ln in lines:
        src = ln.get("en") or ln.get("text") or ln.get("current") or ""
        if mode == "judge":
            out.append({"id": ln["id"], "score": 5, "note": "stub"})
            continue
        if mode in ("fit", "rewrite"):
            words = src.split()
            cap = max(1, int(ln.get("max_syllables", 99) / 1.6))
            src = " ".join(words[:cap])
        elif mode == "baseline":
            src = src + " " + " ".join(src.split()[: max(1, len(src.split()) // 3)])
        out.append({"id": ln["id"], "text": src})
    return {"lines": out}


def env_key(env_name: str) -> str | None:
    return os.environ.get(env_name) if env_name else None

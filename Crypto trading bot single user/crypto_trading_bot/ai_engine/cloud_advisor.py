"""
ai_engine/cloud_advisor.py  —  Multi-Provider Cloud AI Advisor
==============================================================

Supported providers
───────────────────
  deepseek_ollama → local Ollama (handled by deepseek_advisor.py, not here)
  gemini          → Google Gemini (gemini-2.0-flash, etc.)
  openai          → OpenAI (gpt-4o-mini, gpt-4o, etc.)
  anthropic       → Anthropic Claude (claude-3-5-sonnet, etc.)
  groq            → Groq (llama-3.3-70b, free tier, very fast)
  openrouter      → OpenRouter — ONE key accesses ALL models
  universal       → Universal / Custom  — bring your own OpenAI-compat endpoint
                    Use this for: Together.ai, Mistral, Perplexity, Cohere,
                    HuggingFace inference, local vLLM, LM Studio, Ollama
                    (any server that speaks /chat/completions)

RATE LIMIT HANDLING (Gemini 429 fix)
──────────────────────────────────────
Free-tier Gemini allows ~15 RPM.  With 9 pairs firing simultaneously
that limit is exceeded immediately.  Fix:
  • Per-provider token bucket (max 12 calls/min for Gemini free tier)
  • Exponential back-off on 429 (respects Retry-After header)
  • Pairs are serialised through a single asyncio.Semaphore

API keys stored in .env:
  GEMINI_API_KEY
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GROQ_API_KEY
  OPENROUTER_API_KEY
  UNIVERSAL_API_KEY       ← your custom endpoint key
  UNIVERSAL_BASE_URL      ← e.g. https://api.together.xyz/v1
  UNIVERSAL_MODEL         ← e.g. meta-llama/Llama-3-8b-chat-hf
"""

from __future__ import annotations

import os
import re
import time
import asyncio
import logging
from typing import Any

import httpx

from ai_engine.deepseek_advisor import AIVerdict, _build_prediction_prompt, _parse_response

logger = logging.getLogger(__name__)


# ── Provider catalogue ──────────────────────────────────────────────────────

PROVIDER_MODELS: dict[str, list[str]] = {
    "gemini": [
        "gemini-2.0-flash",           # ✓ recommended — fast, free tier
        "gemini-1.5-flash",           # ✓ reliable, free tier
        "gemini-1.5-pro",             # ✓ most capable
        "gemini-1.5-flash-8b",        # ✓ cheapest
        "gemini-1.0-pro",             # ✓ stable fallback
    ],
    "openai": [
        "gpt-4o-mini",                # best value — fast + smart
        "gpt-4o",                     # most capable
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1-mini",                    # reasoning model
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022", # best for analysis
        "claude-3-5-haiku-20241022",  # fast + cheap
        "claude-3-opus-20240229",     # most powerful
        "claude-3-haiku-20240307",
    ],
    "groq": [
        "llama-3.3-70b-versatile",    # free tier ~300 tok/s ← START HERE
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "llama3-70b-8192",
        "gemma2-9b-it",
    ],
    "openrouter": [
        "google/gemini-2.0-flash-exp:free",       # free
        "meta-llama/llama-3.3-70b-instruct:free", # free
        "deepseek/deepseek-r1:free",              # free
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "mistralai/mixtral-8x7b-instruct",
        "deepseek/deepseek-r1",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-flash-1.5",
    ],
    "universal": [
        # Populated from UNIVERSAL_MODEL env var at runtime
        # Examples: together.ai, vLLM, LM Studio, Perplexity, Cohere
        "custom-model",
    ],
}

PROVIDER_BASE_URLS: dict[str, str] = {
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai",
    "openai":     "https://api.openai.com/v1",
    "anthropic":  "https://api.anthropic.com",
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "universal":  "",   # read from UNIVERSAL_BASE_URL env var
}

OPENAI_COMPATIBLE = {"gemini", "openai", "groq", "openrouter", "universal"}

# ── Per-provider rate limits (requests per minute) ──────────────────────────
# Conservative limits — stay safely under free-tier caps
PROVIDER_RPM_LIMIT: dict[str, int] = {
    "gemini":     12,   # free tier = 15 RPM, stay at 12
    "openai":     60,
    "anthropic":  50,
    "groq":       25,   # free tier = 30 RPM
    "openrouter": 20,
    "universal":  60,
}

# Global per-provider semaphore + call timestamps for rate limiting
_provider_locks:    dict[str, asyncio.Semaphore] = {}
_provider_call_ts:  dict[str, list[float]]        = {}


def _get_provider_lock(provider: str) -> asyncio.Semaphore:
    if provider not in _provider_locks:
        _provider_locks[provider] = asyncio.Semaphore(1)
    return _provider_locks[provider]


async def _rate_limit_wait(provider: str) -> None:
    """Block until we're within the provider's RPM limit."""
    rpm = PROVIDER_RPM_LIMIT.get(provider, 60)
    window = 60.0
    now = time.monotonic()

    if provider not in _provider_call_ts:
        _provider_call_ts[provider] = []

    # Remove timestamps older than 1 minute
    _provider_call_ts[provider] = [
        t for t in _provider_call_ts[provider] if now - t < window
    ]

    if len(_provider_call_ts[provider]) >= rpm:
        # Must wait until the oldest call leaves the window
        oldest    = _provider_call_ts[provider][0]
        wait_secs = window - (now - oldest) + 0.1
        logger.info(
            f"[RateLimit] {provider}: {len(_provider_call_ts[provider])}/{rpm} RPM — "
            f"waiting {wait_secs:.1f}s"
        )
        await asyncio.sleep(wait_secs)

    _provider_call_ts[provider].append(time.monotonic())


# ── API key helpers ──────────────────────────────────────────────────────────

def get_api_key(provider: str) -> str:
    env_keys = {
        "gemini":     "GEMINI_API_KEY",
        "openai":     "OPENAI_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
        "groq":       "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "universal":  "UNIVERSAL_API_KEY",
    }
    return os.getenv(env_keys.get(provider, ""), "")


def _get_universal_base_url() -> str:
    url = os.getenv("UNIVERSAL_BASE_URL", "").strip().rstrip("/")
    if not url:
        return ""
    # Ensure it ends with /v1 if missing
    if not url.endswith("/v1") and not url.endswith("/v1/"):
        url = url + "/v1"
    return url


def _get_universal_model() -> str:
    return os.getenv("UNIVERSAL_MODEL", "custom-model").strip()


def list_providers() -> list[dict]:
    result = []
    for provider, models in PROVIDER_MODELS.items():
        key = get_api_key(provider)
        extra = {}
        if provider == "universal":
            extra = {
                "base_url": _get_universal_base_url(),
                "model":    _get_universal_model(),
            }
        result.append({
            "provider":   provider,
            "models":     models,
            "has_key":    bool(key),
            "key_masked": (key[:4] + "••••••••" + key[-4:]) if len(key) >= 8 else ("" if not key else "••••"),
            **extra,
        })
    return result


# ── Main CloudAdvisor ────────────────────────────────────────────────────────

class CloudAdvisor:
    """
    Multi-provider cloud AI advisor with rate limiting and retry.
    Identical public interface to DeepSeekAdvisor.
    """

    def __init__(self, ai_config: dict[str, Any]):
        self.enabled   : bool  = ai_config.get("enabled",        False)
        self.provider  : str   = ai_config.get("provider",       "groq").lower()
        self.model     : str   = ai_config.get("model",          "llama-3.3-70b-versatile")
        self.timeout   : int   = ai_config.get("timeout",        60)
        self.min_conf  : float = ai_config.get("min_confidence", 0.45)
        self.ai_weight : float = ai_config.get("ai_weight",      0.65)
        self.api_key   : str   = get_api_key(self.provider)
        self._healthy  : bool  = True

        # For universal provider, override model from env if not set in config
        if self.provider == "universal":
            env_model = _get_universal_model()
            if env_model and env_model != "custom-model":
                self.model = env_model

    # ── Public ───────────────────────────────────────────────────────────────

    async def analyse(
        self,
        pair:          str,
        base_signal:   dict[str, Any],
        indicators:    dict[str, Any],
        ticker:        dict[str, Any],
        candles:       list[dict] | None = None,
        recent_trades: list[dict] | None = None,
    ) -> AIVerdict:
        if not self.enabled:
            return self._passthrough(base_signal, "Cloud AI disabled")

        if not self.api_key:
            # Universal provider: base URL required too
            if self.provider == "universal":
                missing = []
                if not self.api_key:          missing.append("UNIVERSAL_API_KEY")
                if not _get_universal_base_url(): missing.append("UNIVERSAL_BASE_URL")
                if missing:
                    logger.warning(f"[CloudAI] Universal provider missing: {', '.join(missing)}")
                    return self._passthrough(base_signal, f"Universal: set {', '.join(missing)} in .env")
            else:
                logger.warning(f"[CloudAI] No API key for '{self.provider}'")
                return self._passthrough(base_signal, f"No API key for {self.provider} — add in Settings")

        try:
            # Serialise calls per provider to avoid burst rate-limit hits
            async with _get_provider_lock(self.provider):
                await _rate_limit_wait(self.provider)
                return await self._query_with_retry(pair, base_signal, indicators, ticker, candles or [])
        except Exception as e:
            logger.warning(f"[CloudAI] {self.provider} failed for {pair}: {e}")
            self._healthy = False
            return self._passthrough(base_signal, f"{self.provider} error: {type(e).__name__}")

    async def health_check(self) -> dict[str, Any]:
        key_present = bool(self.api_key)
        base = {
            "provider":    self.provider,
            "model":       self.model,
            "enabled":     self.enabled,
            "has_api_key": key_present,
            "key_masked":  (self.api_key[:4] + "••••••••" + self.api_key[-4:])
                           if len(self.api_key) >= 8 else "",
        }

        if self.provider == "universal":
            bu = _get_universal_base_url()
            base["base_url"] = bu
            if not bu:
                return {**base, "online": False, "error": "Set UNIVERSAL_BASE_URL in .env"}

        if not key_present and self.provider != "universal":
            return {**base, "online": False, "error": "No API key configured"}

        try:
            if self.provider == "anthropic":
                ping_url = "https://api.anthropic.com/v1/models"
                headers  = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
                params   = {}
            elif self.provider == "gemini":
                ping_url = "https://generativelanguage.googleapis.com/v1beta/models"
                headers  = {}
                params   = {"key": self.api_key}
            elif self.provider == "universal":
                bu = _get_universal_base_url()
                ping_url = bu.rstrip("/v1") + "/models" if bu else ""
                headers  = {"Authorization": f"Bearer {self.api_key}"}
                params   = {}
                if not ping_url:
                    return {**base, "online": False, "error": "UNIVERSAL_BASE_URL not set"}
            else:
                ping_url = PROVIDER_BASE_URLS[self.provider] + "/models"
                headers  = {"Authorization": f"Bearer {self.api_key}"}
                params   = {}

            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(ping_url, headers=headers, params=params)
                online = r.status_code < 500

            self._healthy = True
            note = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
            return {**base, "online": online, "note": note}

        except Exception as e:
            self._healthy = False
            return {**base, "online": False, "error": str(e)}

    @property
    def is_healthy(self) -> bool:
        return self._healthy and self.enabled and bool(self.api_key)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _passthrough(self, base_signal: dict, reason: str) -> AIVerdict:
        return AIVerdict(
            action     = base_signal.get("action", "HOLD"),
            confidence = float(base_signal.get("score", 0)),
            score      = float(base_signal.get("score", 0)),
            reasoning  = reason,
            used_ai    = False,
        )

    async def _query_with_retry(
        self,
        pair:        str,
        base_signal: dict,
        indicators:  dict,
        ticker:      dict,
        candles:     list[dict],
        max_retries: int = 3,
    ) -> AIVerdict:
        """Query with exponential back-off retry on 429 / 503."""
        price  = float(ticker.get("last", 0))
        prompt = _build_prediction_prompt(pair, price, candles, indicators, ticker)

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.monotonic()
                if self.provider == "anthropic":
                    text, prompt_tok, comp_tok = await self._query_anthropic(prompt)
                else:
                    text, prompt_tok, comp_tok = await self._query_openai_compat(prompt)
                latency_ms = (time.monotonic() - t0) * 1000

                # Record tokens
                try:
                    from ai_engine import _token_callback
                    if _token_callback:
                        _token_callback(pair, prompt_tok, comp_tok, self.model)
                except Exception:
                    pass

                return self._build_verdict(text, base_signal, price, pair, latency_ms)

            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                # 429 Too Many Requests — back off and retry
                if status == 429:
                    retry_after = float(e.response.headers.get("Retry-After", attempt * 10))
                    wait = min(retry_after, 60.0)
                    logger.warning(
                        f"[CloudAI] {self.provider} 429 on {pair} "
                        f"(attempt {attempt}/{max_retries}) — waiting {wait:.0f}s"
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(wait)
                    last_err = e
                    continue

                # 503 Service Unavailable — short retry
                if status == 503:
                    wait = attempt * 5.0
                    logger.warning(f"[CloudAI] {self.provider} 503 — retry in {wait:.0f}s")
                    if attempt < max_retries:
                        await asyncio.sleep(wait)
                    last_err = e
                    continue

                # Other HTTP errors — log and raise immediately
                try:
                    err_body = e.response.json()
                    err_msg  = err_body.get("error", {}).get("message", e.response.text[:300])
                except Exception:
                    err_msg = e.response.text[:300]
                logger.error(f"[CloudAI] {self.provider} HTTP {status}: {err_msg}")
                raise

            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    await asyncio.sleep(attempt * 2.0)
                continue

        raise last_err or RuntimeError(f"{self.provider} failed after {max_retries} attempts")

    def _build_verdict(
        self,
        text:        str,
        base_signal: dict,
        price:       float,
        pair:        str,
        latency_ms:  float,
    ) -> AIVerdict:
        parsed = _parse_response(text)
        if parsed is None:
            # Log enough to diagnose: show first 600 chars including any code blocks
            preview = text[:600].replace("\n", " ↵ ")
            logger.warning(
                f"[CloudAI] Parse failed for {pair} via {self.provider}. "
                f"Response preview: {preview}"
            )
            return self._passthrough(base_signal, f"{self.provider} parse error — model did not output valid JSON")

        def _f(key: str, default: float) -> float:
            v = parsed.get(key)
            try:    return float(v) if v is not None else default
            except: return default

        ai_action  = str(parsed.get("action",             "HOLD")).upper()
        direction  = str(parsed.get("predicted_direction","SIDEWAYS")).upper()
        sig_conf   = _f("signal_confidence",    _f("confidence", 0.0))
        pred_conf  = _f("prediction_confidence", 0.0)
        ai_score   = _f("confluence_score",      0.0)
        pred_close = _f("predicted_close",  price)
        pred_high  = _f("predicted_high",   price)
        pred_low   = _f("predicted_low",    price)
        support    = _f("support_level",    price * 0.99)
        resistance = _f("resistance_level", price * 1.01)
        reasoning  = str(parsed.get("reasoning", "Analysis complete"))

        logger.info(
            f"[CloudAI] {self.provider}/{self.model} | {pair}: {ai_action} "
            f"dir={direction} close={pred_close:.4f} conf={sig_conf:.2f} {latency_ms:.0f}ms"
        )

        if sig_conf < self.min_conf:
            return AIVerdict(
                action=base_signal.get("action", "HOLD"),
                confidence=sig_conf, score=ai_score,
                reasoning=f"Low confidence ({sig_conf:.0%}) — technical signal used",
                predicted_direction=direction,
                predicted_close=pred_close, predicted_high=pred_high,
                predicted_low=pred_low, support_level=support,
                resistance_level=resistance, prediction_confidence=pred_conf,
                used_ai=True, latency_ms=latency_ms,
            )

        base_score = float(base_signal.get("score", 0.0))
        blended    = (self.ai_weight * ai_score) + ((1 - self.ai_weight) * base_score)
        self._healthy = True

        # Sanity check: if predicted_close == current price, the model didn't
        # actually predict anything (just returned the default). Mark as used_ai=True
        # but flag it so the dashboard can show "weak prediction" styling.
        prediction_is_trivial = abs(pred_close - price) < price * 0.00001 if price > 0 else True

        return AIVerdict(
            action=ai_action, confidence=sig_conf, score=blended,
            reasoning=reasoning + (" [trivial prediction]" if prediction_is_trivial else ""),
            predicted_direction=direction,
            predicted_close=pred_close, predicted_high=pred_high,
            predicted_low=pred_low, support_level=support,
            resistance_level=resistance, prediction_confidence=pred_conf,
            used_ai=True, latency_ms=latency_ms,
        )

    # ── HTTP: OpenAI-compatible ───────────────────────────────────────────────

    async def _query_openai_compat(self, prompt: str) -> tuple[str, int, int]:
        if self.provider == "universal":
            base_url = _get_universal_base_url()
            if not base_url:
                raise ValueError("UNIVERSAL_BASE_URL not set in .env")
        else:
            base_url = PROVIDER_BASE_URLS[self.provider]

        # Auth
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        params: dict = {}

        # Gemini: ALSO send key as query param (some key types need it)
        if self.provider == "gemini":
            params["key"] = self.api_key

        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://cryptobot-pro"
            headers["X-Title"]      = "CryptoBot Pro"

        # System + user messages
        system_content = (
            "You are an expert quantitative crypto trading analyst. "
            "Reason step-by-step, then output a single JSON block inside <answer> tags."
        )

        # Gemini: merge system into user message to avoid role compatibility issues.
        # Do NOT use [INST] tags — those are Llama/Mistral format and confuse Gemini.
        if self.provider == "gemini":
            messages = [{
                "role":    "user",
                "content": f"SYSTEM INSTRUCTION: {system_content}\n\nUSER REQUEST:\n{prompt}",
            }]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": prompt},
            ]

        # Token limits
        max_tok = 1024 if self.provider == "gemini" else 1800

        # Effective timeout
        eff_timeout = min(self.timeout, 30) if self.provider == "gemini" else self.timeout

        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": 0.1,
            "max_tokens":  max_tok,
        }

        async with httpx.AsyncClient(timeout=eff_timeout) as client:
            r = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                params=params,
                json=payload,
            )
            r.raise_for_status()

        data           = r.json()
        content        = data["choices"][0]["message"]["content"]
        usage          = data.get("usage", {})
        prompt_tok     = int(usage.get("prompt_tokens",     0))
        completion_tok = int(usage.get("completion_tokens", 0))
        return content, prompt_tok, completion_tok

    # ── HTTP: Anthropic ───────────────────────────────────────────────────────

    async def _query_anthropic(self, prompt: str) -> tuple[str, int, int]:
        headers = {
            "x-api-key":         self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        payload = {
            "model":      self.model,
            "max_tokens": 1800,
            "system": (
                "You are an expert quantitative crypto trading analyst. "
                "Reason step-by-step, then output a single JSON block inside <answer> tags."
            ),
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()

        data           = r.json()
        blocks         = data.get("content", [])
        content        = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage          = data.get("usage", {})
        prompt_tok     = int(usage.get("input_tokens",  0))
        completion_tok = int(usage.get("output_tokens", 0))
        return content, prompt_tok, completion_tok
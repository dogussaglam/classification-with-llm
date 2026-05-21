"""Thin Groq SDK wrapper with retry, RPD detection, and reasoning_effort plumbing."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import groq

DEFAULT_MAX_TOKENS = 300
REASONING_MAX_TOKENS = 1024
REASONING_MODEL_PREFIXES = ("openai/gpt-oss", "qwen/qwen3")

RETRY_BACKOFFS_SEC: tuple[int, ...] = (2, 5, 15, 45, 180)
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_HONORED_SEC = 300


@dataclass(frozen=True)
class CompletionResult:
    """One Groq chat-completion outcome, including hidden-reasoning bookkeeping."""

    text: str
    reasoning_text: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    model: str
    error: str


class RPDCapHitError(RuntimeError):
    """Raised when Groq returns 429 with Retry-After > 5 min, or after retries fail."""

    def __init__(self, model: str, retry_after_sec: int | None) -> None:
        self.model = model
        self.retry_after_sec = retry_after_sec
        super().__init__(f"RPD cap hit for {model} (retry_after={retry_after_sec}s)")


def _is_reasoning_model(model_slug: str) -> bool:
    return any(model_slug.startswith(p) for p in REASONING_MODEL_PREFIXES)


def _parse_retry_after(exc: groq.RateLimitError) -> int | None:
    """Extract Retry-After from a Groq RateLimitError, in seconds.

    Args:
        exc: The Groq RateLimitError raised by the SDK.

    Returns:
        Seconds to wait (int) or None if the header is absent or unparseable.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _extract_reasoning_tokens(usage) -> int:
    """Read usage.completion_tokens_details.reasoning_tokens if present."""
    if usage is None:
        return 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return 0
    rt = getattr(details, "reasoning_tokens", None)
    if rt is None and isinstance(details, dict):
        rt = details.get("reasoning_tokens")
    return int(rt) if rt else 0


class GroqLabeler:
    """Single-model Groq client with retry + RPD detection.

    Reasoning-capable models (openai/gpt-oss-*, qwen/qwen3-*) receive
    `reasoning_effort` and `reasoning_format='hidden'` (if hide_reasoning=True)
    routed first as direct kwargs and, on TypeError, via extra_body — covering
    both SDK styles (see design §6.6).
    """

    def __init__(
        self,
        model_slug: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        hide_reasoning: bool = True,
        timeout_sec: float = 60.0,
    ) -> None:
        """Create a labeler bound to one Groq model slug.

        Args:
            model_slug: Physical Groq model id (e.g. ``"openai/gpt-oss-120b"``).
            api_key: Override; defaults to ``GROQ_API_KEY`` env var.
            temperature: Sampling temperature; default 0.0 (deterministic).
            max_tokens: Output token cap. Auto-resolves to 1024 for
                reasoning-prefixed slugs, 300 otherwise.
            reasoning_effort: ``"none"`` / ``"default"`` / ``"low"`` / ``"medium"``
                / ``"high"`` for reasoning-capable models; ``None`` for others.
            hide_reasoning: When True (default) and ``reasoning_effort`` is
                set, pass ``reasoning_format="hidden"`` so the response
                ``message.content`` contains only the final answer.
            timeout_sec: Per-request timeout.

        Raises:
            RuntimeError: if ``GROQ_API_KEY`` is missing and ``api_key`` is None.
        """
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY missing. Set it in .env or the environment before running."
            )
        self.model_slug = model_slug
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.reasoning_effort = reasoning_effort
        self.hide_reasoning = hide_reasoning
        self._resolved_max_tokens = max_tokens or (
            REASONING_MAX_TOKENS if _is_reasoning_model(model_slug) else DEFAULT_MAX_TOKENS
        )
        self._sdk = groq.Groq(api_key=key)
        self._direct_kwargs: dict = {}
        self._extra_body: dict = {}
        if reasoning_effort is not None:
            self._direct_kwargs["reasoning_effort"] = reasoning_effort
            if hide_reasoning:
                self._direct_kwargs["reasoning_format"] = "hidden"
        # First call tries direct kwargs; on TypeError we copy to extra_body
        # and remember to skip the direct path going forward.
        self._reasoning_via_extra_body = False

    def _build_call_kwargs(self) -> dict:
        kwargs: dict = {
            "model": self.model_slug,
            "temperature": self.temperature,
            "max_tokens": self._resolved_max_tokens,
            "timeout": self.timeout_sec,
        }
        if self._reasoning_via_extra_body:
            if self._extra_body:
                kwargs["extra_body"] = dict(self._extra_body)
        else:
            kwargs.update(self._direct_kwargs)
        return kwargs

    def _do_request(self, prompt: str):
        kwargs = self._build_call_kwargs()
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._sdk.chat.completions.create(messages=messages, **kwargs)
        except TypeError as e:
            # SDK didn't accept reasoning_effort / reasoning_format as direct
            # kwargs — route through extra_body for this call and all future
            # calls on this client.
            if self._direct_kwargs and not self._reasoning_via_extra_body:
                self._reasoning_via_extra_body = True
                self._extra_body = dict(self._direct_kwargs)
                kwargs = self._build_call_kwargs()
                return self._sdk.chat.completions.create(messages=messages, **kwargs)
            raise e

    def complete(self, prompt: str) -> CompletionResult:
        """Run one completion with retries, returning a CompletionResult.

        Args:
            prompt: Full prompt text (already rendered).

        Returns:
            CompletionResult with text, latency, token counts, and error.

        Raises:
            RPDCapHitError: when Groq signals a daily-cap hit (Retry-After
                exceeds the honored window, or all retries fail with 429).
            groq.APIStatusError: on non-retryable HTTP errors (e.g., 400).
        """
        last_exc: Exception | None = None
        for attempt, backoff in enumerate([0, *RETRY_BACKOFFS_SEC]):
            if backoff:
                time.sleep(backoff)
            t0 = time.perf_counter()
            try:
                resp = self._do_request(prompt)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                return CompletionResult(
                    text=(choice.message.content or ""),
                    reasoning_text=(getattr(choice.message, "reasoning", "") or ""),
                    latency_ms=latency_ms,
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    reasoning_tokens=_extract_reasoning_tokens(usage),
                    model=getattr(resp, "model", self.model_slug),
                    error="",
                )
            except groq.RateLimitError as e:
                retry_after = _parse_retry_after(e)
                if retry_after is not None and retry_after > MAX_RETRY_AFTER_HONORED_SEC:
                    raise RPDCapHitError(self.model_slug, retry_after) from e
                if retry_after is not None:
                    time.sleep(retry_after)
                last_exc = e
                continue
            except groq.APIStatusError as e:
                if getattr(e, "status_code", None) in RETRYABLE_STATUSES:
                    last_exc = e
                    continue
                raise
            except (groq.APIConnectionError, groq.APITimeoutError) as e:
                last_exc = e
                continue
            # Loop exit happens via return on success or via exception otherwise;
            # if we reach here, fall through to the next backoff iteration.
            _ = attempt
        if isinstance(last_exc, groq.RateLimitError):
            raise RPDCapHitError(self.model_slug, None) from last_exc
        if last_exc is None:
            raise RuntimeError(
                f"GroqLabeler.complete: retry loop exited without result for {self.model_slug}"
            )
        raise last_exc

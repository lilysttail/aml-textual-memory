"""A small OpenAI-compatible chat client (step 2): one user message in, one text and the reported token usage out.

Standard library only; the transport is replaceable so everything above it is tested with fake replies. The key is read from
the environment variable named in the config at the moment of the call — it is never stored on the client, never printed and
never put into an error. Errors carry the HTTP status and a short error code only, never the body text.
"""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from . import paid_gate

RETRY_STATUS = (429, 500, 502, 503, 504)
ANTHROPIC_VERSION = "2023-06-01"
ASSUMED_COMPLETION_UPPER_BOUND = 4096          # the most a reply is assumed to write when the request sets no max_tokens (the official judge shape)


class ChatError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class ChatConfig:
    base_url: str                              # up to and including /v1; the path /chat/completions is added
    model: str
    api_key_env: str                           # NAME of the environment variable that holds the key
    temperature: Optional[float] = 0.0         # None = leave it out of the request (a model that refuses any other value)
    max_tokens: Optional[int] = None
    extra_body: dict = field(default_factory=dict)      # merged into the top level of the request body
    headers: dict = field(default_factory=dict)         # extra request headers (a gateway's session id); never the key or the content type
    api_style: str = "openai"                           # "openai" = POST {base}/chat/completions; "anthropic" = POST {base}/messages
    user_agent: str = "agent-memory-challenge-ruler/1"
    timeout_s: float = 60.0
    max_retries: int = 3                       # after the first try; only 429 / 5xx / network errors are retried


@dataclass
class ChatResult:
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    model: Optional[str]
    attempts: int = 1
    finish_reason: Optional[str] = None
    extra: dict = field(default_factory=dict)      # what only some callers have (e.g. tool-call count and cost from a CLI)
    metered: bool = False                          # the client already booked this call in the spending meter (the caller must not again)


Transport = Callable[[str, dict, bytes, float], Tuple[int, bytes]]


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _short_code(payload: bytes) -> Optional[str]:
    """Only a short error code out of an error body (never the message text)."""
    try:
        err = json.loads(payload.decode("utf-8")).get("error")
        code = err.get("code") if isinstance(err, dict) else None
        code = str(code) if code is not None else None
        return code if code and len(code) <= 40 and all(ch.isalnum() or ch in "._-" for ch in code) else None
    except Exception:
        return None


class ChatClient:
    def __init__(self, cfg: ChatConfig, transport: Optional[Transport] = None, sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self._transport = transport or urllib_transport
        self._sleep = sleep
        self._meter = None
        self._price_key = None

    def attach_meter(self, meter, price_key: str) -> None:
        """From now on every attempt goes through the meter's paid gate (a hard wall) and is booked there."""
        self._meter = meter
        self._price_key = price_key

    def __repr__(self) -> str:
        return f"ChatClient(model={self.cfg.model!r}, base_url={self.cfg.base_url!r})"

    def _body(self, prompt: str) -> bytes:
        c = self.cfg
        body: dict = {"model": c.model, "messages": [{"role": "user", "content": prompt}]}
        if c.temperature is not None:
            body["temperature"] = c.temperature
        if c.max_tokens is not None:
            body["max_tokens"] = c.max_tokens
        elif c.api_style == "anthropic":
            body["max_tokens"] = ASSUMED_COMPLETION_UPPER_BOUND         # the messages API insists on one
        body.update(c.extra_body)
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def complete(self, prompt: str) -> ChatResult:
        c = self.cfg
        key = os.environ.get(c.api_key_env, "")
        if not key:
            raise ChatError(f"the environment variable {c.api_key_env} is not set")
        headers = {k: v for k, v in c.headers.items() if k.lower() not in ("authorization", "content-type", "x-api-key")}
        headers.update({"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": c.user_agent})
        if c.api_style == "anthropic":
            headers["x-api-key"] = key
            headers["anthropic-version"] = ANTHROPIC_VERSION
        url = c.base_url.rstrip("/") + ("/messages" if c.api_style == "anthropic" else "/chat/completions")
        body = self._body(prompt)
        attempts = 0
        while True:
            attempts += 1
            try:
                status, payload, result, unreadable = self._attempt(url, headers, body, attempts)
            except (OSError, TimeoutError) as e:
                status, payload, result, unreadable = None, b"", None, False
                err = ChatError(f"network error ({type(e).__name__})")
            else:
                err = None
            if status == 200:
                if unreadable:
                    raise ChatError("the reply could not be read")
                return result
            if err is None:
                err = ChatError(f"HTTP {status}", status=status, code=_short_code(payload))
            if (status is None or status in RETRY_STATUS) and attempts <= c.max_retries:
                self._sleep(min(2.0 * attempts, 10.0))
                continue
            raise err

    def _attempt(self, url, headers, body, attempts):
        """One send. Returns (status, payload, result, unreadable). With a meter attached the send goes through the meter's paid gate:
        only one paid call is outside at any time; it is refused (BudgetExceeded) when 'spent + the most it can cost' passes the limit; what
        it cost is booked inside the same step — the reply's own numbers, or the upper bound when the call raised, the usage is unreadable
        or the 200 reply cannot be parsed; an error reply (not processed) is not booked."""
        c = self.cfg
        metered = self._meter is not None and bool(self._price_key)

        def go():
            status, payload = self._transport(url, headers, body, c.timeout_s)
            if status != 200:
                return (status, payload, None, False), None
            try:
                result = self._parse(payload, attempts)
            except ChatError:
                return (status, payload, None, True), paid_gate.BOUND
            has_usage = result.prompt_tokens is not None and result.completion_tokens is not None
            result.metered = metered
            return (status, payload, result, False), ((result.prompt_tokens, result.completion_tokens) if has_usage else paid_gate.BOUND)

        if not metered:
            return go()[0]
        upper_completion = c.max_tokens if c.max_tokens is not None else ASSUMED_COMPLETION_UPPER_BOUND
        return self._meter.gate.run(self._price_key, len(body), upper_completion, go)

    def _parse(self, payload: bytes, attempts: int) -> ChatResult:
        if self.cfg.api_style == "anthropic":
            return self._parse_messages(payload, attempts)
        try:
            doc = json.loads(payload.decode("utf-8"))
            choice = doc["choices"][0]
            text = choice["message"].get("content")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ChatError("the reply could not be read")
        usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
        return ChatResult(text="" if text is None else str(text), prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"), model=doc.get("model"), attempts=attempts,
                          finish_reason=choice.get("finish_reason"))

    @staticmethod
    def _parse_messages(payload: bytes, attempts: int) -> ChatResult:
        """A messages-style reply: the text blocks joined (thinking and other blocks are not part of the answer)."""
        try:
            doc = json.loads(payload.decode("utf-8"))
            blocks = doc["content"]
            if not isinstance(blocks, list):
                raise TypeError("content is not a list")
            text = "".join(b["text"] for b in blocks if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str))
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ChatError("the reply could not be read")
        usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
        return ChatResult(text=text, prompt_tokens=usage.get("input_tokens"), completion_tokens=usage.get("output_tokens"),
                          model=doc.get("model"), attempts=attempts, finish_reason=doc.get("stop_reason"))

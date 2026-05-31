"""Qwen/OpenAI-compatible chat client configuration.

This module intentionally uses a tiny stdlib HTTP client instead of the OpenAI
Python SDK so Qwen/Modal failures do not surface as `openai.*` exceptions.
For Qwen/DashScope, set `QWEN_API_KEY` (or `DASHSCOPE_API_KEY`) and optionally
`QWEN_BASE_URL` / `QWEN_MODEL`.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
from urllib import error, request
from urllib.parse import urlparse


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class ChatClientConfig:
    model: str
    api_key_env: str
    base_url: Optional[str]


def resolve_chat_config(model: Optional[str] = None, role: str = "AGENT") -> ChatClientConfig:
    """Resolve model/API settings from env with Qwen as a first-class backend."""
    from dotenv import load_dotenv

    load_dotenv()
    role_model = os.environ.get(f"{role.upper()}_MODEL")
    generic_model = os.environ.get("LLM_MODEL")
    qwen_model = os.environ.get("QWEN_MODEL")

    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if qwen_key:
        return ChatClientConfig(
            model=model or role_model or generic_model or qwen_model or DEFAULT_QWEN_MODEL,
            api_key_env="QWEN_API_KEY" if os.environ.get("QWEN_API_KEY") else "DASHSCOPE_API_KEY",
            base_url=(
                os.environ.get("QWEN_BASE_URL")
                or os.environ.get("LLM_BASE_URL")
                or DEFAULT_QWEN_BASE_URL
            ),
        )

    if openai_key:
        return ChatClientConfig(
            model=model or role_model or generic_model or DEFAULT_OPENAI_MODEL,
            api_key_env="OPENAI_API_KEY",
            base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL"),
        )

    raise RuntimeError(
        "No chat model API key is set. For Qwen, set QWEN_API_KEY or "
        "DASHSCOPE_API_KEY in .env. For OpenAI-compatible alternatives, set "
        "OPENAI_API_KEY plus OPENAI_BASE_URL/LLM_BASE_URL as needed."
    )


def make_openai_compatible_client(
    model: Optional[str] = None,
    role: str = "AGENT",
    max_retries: int = 6,
    request_timeout: float = 30.0,
):
    """Create a direct HTTP client for Qwen/OpenAI-compatible chat endpoints."""
    config = resolve_chat_config(model=model, role=role)
    return ChatClient(
        api_key=os.environ[config.api_key_env],
        base_url=config.base_url or DEFAULT_OPENAI_BASE_URL,
        timeout=request_timeout,
        max_retries=max_retries,
    ), config


class ChatClient:
    """Minimal `client.chat.completions.create(...)` compatible client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 6,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_chat_completion)
        )

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: Optional[dict[str, Any]] = None,
    ):
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._post_chat_completion(payload)
            except RuntimeError:
                raise
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"chat completion failed after retries: {last_error}") from last_error

    def _post_chat_completion(self, payload: dict[str, Any]):
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            # Modal root URLs need `/v1`; retry once with that suffix to turn a
            # common configuration mistake into a working request.
            if e.code == 404 and not _base_url_has_v1(self.base_url):
                return self._post_chat_completion_with_base(
                    f"{self.base_url}/v1", payload
                )
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"chat backend returned HTTP {e.code} for {url}: {detail}"
            ) from e
        except TimeoutError as e:
            raise TimeoutError(
                f"chat backend timed out after {self.timeout}s at {url}"
            ) from e
        return _to_completion_response(data)

    def _post_chat_completion_with_base(self, base_url: str, payload: dict[str, Any]):
        original = self.base_url
        try:
            self.base_url = base_url.rstrip("/")
            return self._post_chat_completion(payload)
        finally:
            self.base_url = original


def _base_url_has_v1(base_url: str) -> bool:
    return urlparse(base_url).path.rstrip("/").endswith("/v1")


def _to_completion_response(data: dict[str, Any]):
    try:
        content = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected chat completion response: {data!r}") from e
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content))
        ]
    )

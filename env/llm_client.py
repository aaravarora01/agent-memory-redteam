"""OpenAI-compatible chat client configuration.

The OpenAI Python SDK is used as a transport for any provider that implements
the OpenAI chat-completions API. For Qwen/DashScope, set `QWEN_API_KEY` (or
`DASHSCOPE_API_KEY`) and optionally `QWEN_BASE_URL` / `QWEN_MODEL`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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
    """Create an OpenAI SDK client for Qwen/OpenAI-compatible chat endpoints."""
    import openai

    config = resolve_chat_config(model=model, role=role)
    api_key = os.environ[config.api_key_env]
    kwargs = {
        "api_key": api_key,
        "max_retries": max_retries,
        "timeout": request_timeout,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return openai.OpenAI(**kwargs), config

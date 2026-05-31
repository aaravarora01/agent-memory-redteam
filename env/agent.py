"""LLM-backed agent (plan §1.4).

`Agent.act(query, retrieved_memories)` calls an OpenAI-compatible chat endpoint
with a system prompt saying "use the user's memory to answer." Qwen/DashScope is
supported via `QWEN_API_KEY` / `DASHSCOPE_API_KEY`.
"""

from __future__ import annotations

from typing import List, Optional

from .memory_store import MemoryEntry


DEFAULT_SYSTEM_PROMPT = (
    "You are the user's personal AI assistant. You have access to entries from "
    "the user's long-term memory store, retrieved for the current query. Use "
    "the retrieved memory entries to ground your answer in what the user has "
    "actually said, done, or preferred. Keep answers concise (1-3 sentences) "
    "and directly answer the user's question."
)


def _format_memories(retrieved: List[MemoryEntry]) -> str:
    if not retrieved:
        return "(no memory entries retrieved)"
    lines = []
    for i, m in enumerate(retrieved, start=1):
        lines.append(f"[{i}] {m.text}")
    return "\n".join(lines)


class Agent:
    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 256,
        client=None,
        max_retries: int = 6,
        request_timeout: float = 30.0,
    ):
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client
        # SDK-level retry handles 429s + transient 5xx with exponential backoff.
        # Default is 2; bump for long sweeps (Exp 1 = ~1600 episodes).
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    @property
    def client(self):
        if self._client is None:
            from .llm_client import make_openai_compatible_client

            self._client, config = make_openai_compatible_client(
                model=self.model,
                role="AGENT",
                max_retries=self.max_retries,
                request_timeout=self.request_timeout,
            )
            self.model = config.model
        return self._client

    def build_user_message(self, query: str, retrieved: List[MemoryEntry]) -> str:
        return (
            "Retrieved memory entries:\n"
            f"{_format_memories(retrieved)}\n\n"
            f"User query: {query}"
        )

    def act(self, query: str, retrieved: List[MemoryEntry]) -> str:
        user_msg = self.build_user_message(query, retrieved)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

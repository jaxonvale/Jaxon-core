"""Thin wrapper around the Anthropic SDK used by the CEO and department agents."""

from __future__ import annotations

from typing import Any

from .config import Config

FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}


class LLM:
    def __init__(self, config: Config, client: Any = None):
        self.config = config
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def call(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             effort: str | None = None) -> Any:
        """One streamed Messages request; returns the final message."""
        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive"},
            output_config={"effort": effort or self.config.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.config.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        with self.client.beta.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

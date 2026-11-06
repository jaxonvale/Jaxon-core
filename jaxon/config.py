"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_db() -> str:
    return os.environ.get("JAXON_DB", str(Path.home() / ".jaxon" / "jaxon.db"))


@dataclass
class Config:
    model: str = field(default_factory=lambda: os.environ.get("JAXON_MODEL", "claude-opus-5-5"))
    # Effort for the CEO's own reasoning and for department heads.
    effort: str = field(default_factory=lambda: os.environ.get("JAXON_EFFORT", "high"))
    department_effort: str = field(default_factory=lambda: os.environ.get("JAXON_DEPT_EFFORT", "medium"))
    max_tokens: int = 64000
    max_tool_rounds: int = 25
    db_path: str = field(default_factory=_default_db)
    web_search: bool = field(default_factory=lambda: os.environ.get("JAXON_WEB_SEARCH", "1") != "0")
    # Server-side refusal fallback (Claude API only). Set JAXON_FALLBACKS=0 on other platforms.
    fallbacks: bool = field(default_factory=lambda: os.environ.get("JAXON_FALLBACKS", "1") != "0")

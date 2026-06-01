# Copyright (c) Opendatalab. All rights reserved.
"""
Central configuration for LightOnOCR backends.

Supported environment variables (in priority order):

  LLM_SERVICE          Service type: "openai" / "azure" / "anthropic" / etc.
                       Use "local" to skip API and run model locally.
                       Default: "openai"

  OPENAI_API_KEY       API key for OpenAI-compatible endpoint.
                       For LM Studio: "lm-studio"
                       For real OpenAI: "sk-..."
                       Default: ""  (no auth needed for local servers)

  OPENAI_API_BASE      Base URL of the API server (without path suffix).
                       Example: "http://localhost:1234/v1"
                       Example: "https://api.openai.com/v1"
                       Default: "http://localhost:1234/v1"

  OPENAI_MODEL         Model name to use.
                       Default: "lightonocr"

  LOCAL_MODEL_ID       HuggingFace model id for local inference.
                       Default: "mlx-community/LightOnOCR-2-1B-bf16"

  LOCAL_MODEL_BACKEND  "mlx" | "transformers"
                       Default: "mlx" on Apple Silicon, else "transformers"

Backward-compatible (still work, lower priority):

  LIGHTON_SERVER_URL   Full URL to /chat/completions endpoint.
  LIGHTON_MODEL_NAME   Model name.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from loguru import logger


# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_API_BASE  = "http://localhost:1234/v1"
_DEFAULT_MODEL     = "lightonocr"
_DEFAULT_LOCAL_ID  = "mlx-community/LightOnOCR-2-1B-bf16"

def _default_local_backend() -> str:
    """Auto-detect best local backend for the current machine."""
    is_apple_silicon = (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
    )
    return "mlx" if is_apple_silicon else "transformers"


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class LightOnConfig:
    """Resolved configuration for LightOnOCR."""

    # Which service / provider to use
    llm_service: str = "openai"   # e.g. "openai", "azure", "local"

    # API settings
    api_key: str  = ""
    api_base: str = _DEFAULT_API_BASE
    model: str    = _DEFAULT_MODEL

    # Derived full URL (api_base + /chat/completions)
    chat_completions_url: str = field(init=False)

    # Local model settings
    local_model_id: str      = _DEFAULT_LOCAL_ID
    local_backend: str       = field(default_factory=_default_local_backend)

    # Operational flags
    use_api: bool   = True
    use_local: bool = False   # fallback or primary local

    def __post_init__(self):
        base = self.api_base.rstrip("/")
        self.chat_completions_url = f"{base}/chat/completions"

    def __repr__(self) -> str:
        key_preview = (self.api_key[:8] + "...") if len(self.api_key) > 8 else self.api_key or "(empty)"
        return (
            f"LightOnConfig("
            f"service={self.llm_service!r}, "
            f"api_base={self.api_base!r}, "
            f"model={self.model!r}, "
            f"api_key={key_preview!r}, "
            f"use_api={self.use_api}, "
            f"use_local={self.use_local}, "
            f"local_backend={self.local_backend!r}"
            f")"
        )


# ── Loader ────────────────────────────────────────────────────────────────────

def get_lighton_config() -> LightOnConfig:
    """
    Read environment variables and return a resolved LightOnConfig.

    Priority:
      1. New-style vars: LLM_SERVICE, OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL
      2. Legacy vars: LIGHTON_SERVER_URL, LIGHTON_MODEL_NAME
      3. Built-in defaults
    """
    llm_service = os.getenv("LLM_SERVICE", "openai").strip().lower()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    # ── API base URL ──────────────────────────────────────────────────────────
    api_base = os.getenv("OPENAI_API_BASE", "").strip()
    if not api_base:
        # Legacy fallback: strip /chat/completions suffix if present
        legacy_url = os.getenv("LIGHTON_SERVER_URL", "").strip()
        if legacy_url:
            api_base = legacy_url.rstrip("/")
            if api_base.endswith("/chat/completions"):
                api_base = api_base[: -len("/chat/completions")]
    if not api_base:
        api_base = _DEFAULT_API_BASE

    # ── Model name ────────────────────────────────────────────────────────────
    model = (
        os.getenv("OPENAI_MODEL", "").strip()
        or os.getenv("LIGHTON_MODEL_NAME", "").strip()
        or _DEFAULT_MODEL
    )

    # ── Local model ───────────────────────────────────────────────────────────
    local_model_id = os.getenv("LOCAL_MODEL_ID", _DEFAULT_LOCAL_ID).strip()
    local_backend  = os.getenv("LOCAL_MODEL_BACKEND", _default_local_backend()).strip().lower()

    # ── Resolve use_api / use_local ───────────────────────────────────────────
    # "local" → only local, no API
    # anything else → try API first, fallback to local
    use_api   = llm_service != "local"
    use_local = True   # always enable local as fallback (or primary if service==local)

    cfg = LightOnConfig(
        llm_service=llm_service,
        api_key=api_key,
        api_base=api_base,
        model=model,
        local_model_id=local_model_id,
        local_backend=local_backend,
        use_api=use_api,
        use_local=use_local,
    )
    logger.debug(f"LightOnConfig resolved: {cfg}")
    return cfg


# ── Headers helper ────────────────────────────────────────────────────────────

def build_api_headers(cfg: LightOnConfig) -> dict:
    """Return HTTP headers for the configured API service."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return headers

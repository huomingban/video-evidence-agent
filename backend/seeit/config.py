"""Application paths and environment-backed runtime configuration."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "agent.sqlite3"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
LOCAL_QDRANT_DIR = PROJECT_ROOT / "data" / "qdrant"
ENV_PATH = PROJECT_ROOT / "backend" / ".env"

for directory in (DB_PATH.parent, UPLOADS_DIR, LOCAL_QDRANT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if load_dotenv is not None:
    load_dotenv(ENV_PATH)

logger = logging.getLogger("seeit")


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def kimi_settings() -> dict[str, Any]:
    return {
        "api_key": os.getenv("KIMI_API_KEY", "").strip(),
        "base_url": os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1").strip(),
        "model": os.getenv("KIMI_MODEL", "moonshot-v1-8k").strip(),
        "enabled": env_flag("KIMI_ENABLED", True),
        "timeout": float(os.getenv("KIMI_TIMEOUT_SECONDS", "45")),
        "temperature": float(os.getenv("KIMI_TEMPERATURE", "0.6")),
        "thinking": env_flag("KIMI_THINKING", False),
        "trust_env": env_flag("KIMI_TRUST_ENV", False),
        "proxy": os.getenv("KIMI_PROXY", "").strip() or None,
    }


@contextmanager
def direct_connection_if_configured():
    """Temporarily bypass broken system proxies during model downloads."""
    if env_flag("WHISPER_TRUST_ENV", False):
        yield
        return
    proxy_names = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    )
    saved = {name: os.environ.pop(name, None) for name in proxy_names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value

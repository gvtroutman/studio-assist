"""Optional machine-specific defaults, kept outside version control.

Environment variables still take precedence at each caller. This file only holds
endpoint defaults; credentials belong in the existing credential settings.
"""
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "studio.local.json"


def default(name, fallback):
    """Read a nonempty string default without logging private configuration."""
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    value = data.get(name) if isinstance(data, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else fallback

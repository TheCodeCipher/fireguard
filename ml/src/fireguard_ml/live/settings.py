"""Secrets for the live scripts, read from the same ``.env`` files the backend uses.

Order (first one wins): the process environment, ``backend/.env``, the repository's ``.env``.
Keys used here: ``FIRMS_MAP_KEY`` (NASA FIRMS) and ``EARTHENGINE_PROJECT`` (Google Earth Engine
Cloud project). ``.env`` files are git-ignored; never commit a key.
"""

from __future__ import annotations

import os
from pathlib import Path

KEYS = ("FIRMS_MAP_KEY", "EARTHENGINE_PROJECT")


def _parse(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_live_settings(repo_root: Path) -> dict[str, str]:
    """Put FIRMS_MAP_KEY / EARTHENGINE_PROJECT into os.environ if a .env file holds them."""
    found = {}
    for p in (Path(repo_root) / "backend" / ".env", Path(repo_root) / ".env"):
        if p.is_file():
            for k, v in _parse(p).items():
                if k in KEYS and v and k not in found:
                    found[k] = v
    for k, v in found.items():
        os.environ.setdefault(k, v)
    return {k: ("set" if os.environ.get(k) else "missing") for k in KEYS}

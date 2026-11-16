"""Load secrets from the repo-root .env file (see .env.example).

Variables already set in the real environment always win (override=False), so CI,
Colab and shell exports take precedence over the local file. If python-dotenv isn't
installed, this is a no-op.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: str | Path | None = None) -> bool:
    """Load `path` (default: <repo>/.env). Returns True if a file was loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.is_file():
        return False
    return load_dotenv(env_path, override=False)

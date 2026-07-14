import logging
import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Minimal KEY=VALUE loader; real environment variables always win."""
    env_file = path or Path(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str | None
    log_level: str

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file)
        url = os.environ.get("QDRANT_URL")
        if not url:
            raise RuntimeError("QDRANT_URL is not set (use ':memory:' for a local smoke run)")
        return cls(
            qdrant_url=url,
            qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

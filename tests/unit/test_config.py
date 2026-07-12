import os
from pathlib import Path

import pytest

from atlas_core.config import Settings, load_dotenv


def test_load_dotenv_parses_and_defers_to_real_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nQDRANT_URL=\"http://from-file\"\nQDRANT_API_KEY='secret'\nnot a kv line\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "from-real-env")

    load_dotenv(env_file)

    assert os.environ["QDRANT_URL"] == "http://from-file"
    assert os.environ["QDRANT_API_KEY"] == "from-real-env"


def test_load_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    load_dotenv(tmp_path / "absent.env")


def test_settings_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    settings = Settings.from_env(tmp_path / "absent.env")

    assert settings.qdrant_url == ":memory:"
    assert settings.qdrant_api_key is None
    assert settings.log_level == "INFO"


def test_settings_requires_qdrant_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        Settings.from_env(tmp_path / "absent.env")

import os

import pytest

pytest.importorskip("dotenv")

from text2sql.env import load_env  # noqa: E402


def test_loads_missing_vars_without_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("T2S_TEST_NEW=from_file\nT2S_TEST_EXISTING=from_file\n", encoding="utf-8")
    monkeypatch.delenv("T2S_TEST_NEW", raising=False)
    monkeypatch.setenv("T2S_TEST_EXISTING", "from_env")

    assert load_env(env_file) is True
    assert os.environ["T2S_TEST_NEW"] == "from_file"
    assert os.environ["T2S_TEST_EXISTING"] == "from_env"  # real env wins
    monkeypatch.delenv("T2S_TEST_NEW")


def test_missing_file_is_noop(tmp_path):
    assert load_env(tmp_path / "nope.env") is False

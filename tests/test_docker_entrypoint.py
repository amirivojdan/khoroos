"""Exercise container startup decisions without Docker, network access, or model weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker/entrypoint.sh"


@pytest.fixture
def entrypoint(tmp_path):
    log = tmp_path / "calls.jsonl"
    command = tmp_path / "khoroos"
    command.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_CALL_LOG'], 'a') as f:\n"
        "    f.write(json.dumps({'args': sys.argv[1:], "
        "'token_matches': os.environ.get('HF_TOKEN') == 'test-read-token'}) + '\\n')\n"
        "if sys.argv[1:] == ['models', 'download']:\n"
        "    sys.exit(int(os.environ.get('TEST_DOWNLOAD_EXIT', '0')))\n"
    )
    command.chmod(0o755)

    def run(*args, **overrides):
        env = {
            "PATH": f"{tmp_path}:{os.defpath}",
            "TEST_CALL_LOG": str(log),
            **overrides,
        }
        result = subprocess.run(
            ["sh", str(ENTRYPOINT), *args], env=env, capture_output=True, text=True,
            timeout=10,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize("args,prefetch", [
    (["ui"], True),
    (["analyze", "folder with spaces/farm.mp4"], True),
    (["models", "download"], False),
    (["models", "status"], False),
    (["info"], False),
    (["--help"], False),
    (["ui", "--help"], False),
    (["analyze", "--help"], False),
])
def test_auto_prefetch_only_for_inference(entrypoint, args, prefetch):
    result, calls = entrypoint("khoroos", *args)
    assert result.returncode == 0, result.stderr
    expected = [["models", "download"], args] if prefetch else [args]
    assert [call["args"] for call in calls] == expected


@pytest.mark.parametrize("mode,args,expected", [
    ("0", ["ui"], [["ui"]]),
    ("false", ["analyze", "farm.mp4"], [["analyze", "farm.mp4"]]),
    ("1", ["info"], [["models", "download"], ["info"]]),
    ("TRUE", ["info"], [["models", "download"], ["info"]]),
])
def test_prefetch_override(entrypoint, mode, args, expected):
    result, calls = entrypoint("khoroos", *args, KHOROOS_PREFETCH_MODELS=mode)
    assert result.returncode == 0, result.stderr
    assert [call["args"] for call in calls] == expected


def test_download_failure_prevents_server_start(entrypoint):
    result, calls = entrypoint("khoroos", "ui", TEST_DOWNLOAD_EXIT="7")
    assert result.returncode == 7
    assert [call["args"] for call in calls] == [["models", "download"]]


def test_token_file_overrides_environment_without_printing_token(entrypoint, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_bytes(b"test-read-token\r\n")
    result, calls = entrypoint(
        "khoroos", "ui", HF_TOKEN_FILE=str(token_file), HF_TOKEN="old-token",
    )
    assert result.returncode == 0, result.stderr
    assert all(call["token_matches"] for call in calls)
    assert "test-read-token" not in result.stdout + result.stderr


@pytest.mark.parametrize("exists", [True, False])
def test_invalid_token_file_stops_startup(entrypoint, tmp_path, exists):
    token_file = tmp_path / "token"
    if exists:
        token_file.write_text("\n")
    result, calls = entrypoint("khoroos", "ui", HF_TOKEN_FILE=str(token_file))
    assert result.returncode == 1
    assert "HF_TOKEN_FILE" in result.stderr
    assert calls == []


def test_invalid_prefetch_value_stops_startup(entrypoint):
    result, calls = entrypoint("khoroos", "ui", KHOROOS_PREFETCH_MODELS="sometimes")
    assert result.returncode == 2
    assert "auto or a boolean" in result.stderr
    assert calls == []


def test_missing_command_has_clear_error(entrypoint):
    result, calls = entrypoint()
    assert result.returncode == 2
    assert "command is required" in result.stderr
    assert calls == []

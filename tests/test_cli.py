"""CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from greenai import __version__
from greenai.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Green AI Profiler" in result.stdout or "profile" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_system_info() -> None:
    result = runner.invoke(app, ["system-info"])
    assert result.exit_code == 0
    assert "CPU" in result.stdout
    assert "Python" in result.stdout

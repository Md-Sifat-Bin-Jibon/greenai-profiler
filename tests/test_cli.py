"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from greenai import __version__
from greenai.cli import app
from greenai.exceptions import ModelLoadError

runner = CliRunner()
torch = pytest.importorskip("torch")


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


def _tiny_sequential() -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(8, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    model.eval()
    return model


def _failure_text(result: object) -> str:
    exception = getattr(result, "exception", None)
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return f"{stdout}\n{stderr}\n{exception}"


def test_inspect_refuses_pickled_module_without_allow_pickle(tmp_path: Path) -> None:
    path = tmp_path / "module.pt"
    torch.save(_tiny_sequential(), path)
    result = runner.invoke(app, ["inspect", str(path), "--input-shape", "8"])
    assert result.exit_code != 0
    text = _failure_text(result).lower()
    assert "allow-pickle" in text or isinstance(result.exception, ModelLoadError)


def test_inspect_state_dict_explains_architecture_requirement(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    torch.save(_tiny_sequential().state_dict(), path)
    result = runner.invoke(app, ["inspect", str(path), "--input-shape", "8"])
    assert result.exit_code != 0
    text = _failure_text(result).lower()
    assert "state_dict" in text
    assert "allow-pickle will not help" in text or "architecture" in text


def test_benchmark_and_profile_pickled_module(tmp_path: Path) -> None:
    path = tmp_path / "module.pt"
    torch.save(_tiny_sequential(), path)
    bench = runner.invoke(
        app,
        [
            "benchmark",
            str(path),
            "--allow-pickle",
            "--input-shape",
            "8",
            "--warmup",
            "1",
            "--iterations",
            "2",
        ],
    )
    assert bench.exit_code == 0, bench.stdout
    assert (
        "Latency" in bench.stdout or "ms" in bench.stdout.lower() or "mean" in bench.stdout.lower()
    )

    json_out = tmp_path / "profile.json"
    csv_out = tmp_path / "profile.csv"
    profile = runner.invoke(
        app,
        [
            "profile",
            str(path),
            "--allow-pickle",
            "--input-shape",
            "8",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--no-layers",
            "--output",
            str(json_out),
        ],
    )
    assert profile.exit_code == 0, profile.stdout
    assert json_out.is_file()
    assert "schema_version" in json_out.read_text(encoding="utf-8")

    csv_result = runner.invoke(
        app,
        [
            "profile",
            str(path),
            "--allow-pickle",
            "--input-shape",
            "8",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--no-layers",
            "--output",
            str(csv_out),
            "--format",
            "csv",
        ],
    )
    assert csv_result.exit_code == 0, csv_result.stdout
    assert csv_out.is_file()
    assert csv_out.read_text(encoding="utf-8").strip()


def test_compare_and_report_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "module.pt"
    torch.save(_tiny_sequential(), path)
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    for dest in (baseline, optimized):
        result = runner.invoke(
            app,
            [
                "profile",
                str(path),
                "--allow-pickle",
                "--input-shape",
                "8",
                "--warmup",
                "1",
                "--iterations",
                "2",
                "--no-layers",
                "--output",
                str(dest),
            ],
        )
        assert result.exit_code == 0, result.stdout

    compared = runner.invoke(app, ["compare", str(baseline), str(optimized)])
    assert compared.exit_code == 0, compared.stdout

    html_out = tmp_path / "report.html"
    reported = runner.invoke(app, ["report", str(baseline), "--html", str(html_out)])
    assert reported.exit_code == 0, reported.stdout
    assert html_out.is_file()
    assert "html" in html_out.read_text(encoding="utf-8").lower()

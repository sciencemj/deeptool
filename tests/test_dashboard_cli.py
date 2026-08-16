import sys

import pytest

from deeptool import dashboard


def test_dashboard_cli_uses_safe_defaults(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        dashboard.importlib.util, "find_spec", lambda name: object()
    )
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda command, check: captured.update(command=command, check=check),
    )

    assert dashboard.main([str(tmp_path)]) == 0

    command = captured["command"]
    assert command[:3] == [sys.executable, "-m", "streamlit"]
    assert "--server.address=127.0.0.1" in command
    assert "--server.port=8501" in command
    assert command[-3:] == [str(tmp_path.resolve()), "--refresh", "2.0"]
    assert captured["check"] is True


def test_dashboard_cli_forwards_remote_options(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        dashboard.importlib.util, "find_spec", lambda name: object()
    )
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda command, check: captured.update(command=command),
    )

    dashboard.main([
        str(tmp_path),
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--refresh",
        "5",
        "--no-browser",
    ])

    assert "--server.address=0.0.0.0" in captured["command"]
    assert "--server.port=9000" in captured["command"]
    assert "--server.headless=true" in captured["command"]


def test_dashboard_cli_reports_missing_extra(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dashboard.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit) as error:
        dashboard.main([str(tmp_path)])

    assert error.value.code == 2
    assert 'uv add "deeptool[dashboard]"' in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--host", ""],
        ["--port", "0"],
        ["--port", "65536"],
        ["--refresh", "-1"],
        ["--refresh", "nan"],
        ["--refresh", "inf"],
    ],
)
def test_dashboard_cli_rejects_invalid_network_and_refresh_options(
    tmp_path, arguments
):
    with pytest.raises(SystemExit) as error:
        dashboard.main([str(tmp_path), *arguments])
    assert error.value.code == 2


def test_dashboard_cli_rejects_missing_and_file_roots(tmp_path):
    file_root = tmp_path / "history.jsonl"
    file_root.write_text("")

    for root in (tmp_path / "missing", file_root):
        with pytest.raises(SystemExit) as error:
            dashboard.main([str(root)])
        assert error.value.code == 2

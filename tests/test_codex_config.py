from __future__ import annotations

import tomllib

import pytest

from kaggriculture_lab import dashboard
from kaggriculture_lab.dashboard import server as dashboard_server
from kaggriculture_lab.codex_config import (
    MANAGED_END,
    MANAGED_START,
    CodexConfigConflict,
    dashboard_mcp_url,
    sync_codex_mcp_config,
)


def test_sync_codex_mcp_config_creates_project_config(tmp_path):
    destination = sync_codex_mcp_config(
        "http://127.0.0.1:8123/mcp", project_root=tmp_path
    )

    document = tomllib.loads(destination.read_text(encoding="utf-8"))
    server = document["mcp_servers"]["kaggriculture_progress"]
    assert server == {
        "url": "http://127.0.0.1:8123/mcp",
        "enabled": True,
        "required": False,
        "tool_timeout_sec": 30,
        "default_tools_approval_mode": "auto",
    }


def test_sync_updates_only_managed_block(tmp_path):
    config_dir = tmp_path / ".codex"
    config_dir.mkdir()
    config = config_dir / "config.toml"
    config.write_text(
        'model = "gpt-5"\n\n'
        f"{MANAGED_START}\n"
        "[mcp_servers.kaggriculture_progress]\n"
        'url = "http://127.0.0.1:8000/mcp"\n'
        f"{MANAGED_END}\n\n"
        "[features]\nexperimental = true\n",
        encoding="utf-8",
    )

    sync_codex_mcp_config("http://127.0.0.1:9000/mcp", project_root=tmp_path)

    text = config.read_text(encoding="utf-8")
    document = tomllib.loads(text)
    assert text.count(MANAGED_START) == 1
    assert document["model"] == "gpt-5"
    assert document["features"]["experimental"] is True
    assert (
        document["mcp_servers"]["kaggriculture_progress"]["url"]
        == "http://127.0.0.1:9000/mcp"
    )


def test_sync_replaces_old_per_run_servers_with_single_shared_endpoint(tmp_path):
    config_dir = tmp_path / ".codex"
    config_dir.mkdir()
    config = config_dir / "config.toml"
    config.write_text(
        f"{MANAGED_START}\n"
        "[mcp_servers.kaggriculture_progress_fixed]\n"
        'url = "http://127.0.0.1:8000/mcp"\n\n'
        "[mcp_servers.kaggriculture_progress_mixed]\n"
        'url = "http://127.0.0.1:8123/mcp"\n'
        f"{MANAGED_END}\n",
        encoding="utf-8",
    )
    destination = sync_codex_mcp_config(
        "http://127.0.0.1:9000/mcp", project_root=tmp_path
    )

    servers = tomllib.loads(destination.read_text(encoding="utf-8"))["mcp_servers"]
    assert list(servers) == ["kaggriculture_progress"]
    assert servers["kaggriculture_progress"]["url"] == "http://127.0.0.1:9000/mcp"
    assert destination.read_text(encoding="utf-8").count(MANAGED_START) == 1


def test_sync_preserves_unmanaged_same_named_server(tmp_path):
    config_dir = tmp_path / ".codex"
    config_dir.mkdir()
    config = config_dir / "config.toml"
    original = (
        "[mcp_servers.kaggriculture_progress]\n"
        'url = "https://example.internal/mcp"\n'
    )
    config.write_text(original, encoding="utf-8")

    with pytest.raises(CodexConfigConflict, match="unmanaged"):
        sync_codex_mcp_config("http://127.0.0.1:8000/mcp", project_root=tmp_path)

    assert config.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "http://127.0.0.1:8123/mcp"),
        ("::", "http://127.0.0.1:8123/mcp"),
        ("::1", "http://[::1]:8123/mcp"),
        ("localhost", "http://localhost:8123/mcp"),
    ],
)
def test_dashboard_mcp_url_is_client_connectable(host, expected):
    assert dashboard_mcp_url(host, 8123) == expected


def test_standalone_dashboard_registers_its_actual_port(monkeypatch, tmp_path, capsys):
    calls = []

    class FakeServer:
        server_address = ("127.0.0.1", 8432)
        database_available_at_startup = True

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            calls.append("close")

    monkeypatch.chdir(tmp_path)
    # `serve` lives in `dashboard.server` and resolves both of these from that
    # module's globals, so patching the package facade would be silently ignored.
    monkeypatch.setattr(dashboard_server, "make_server", lambda *args: FakeServer())
    monkeypatch.setattr(
        dashboard_server,
        "sync_codex_mcp_config",
        lambda url: calls.append(url)
        or tmp_path / ".codex" / "config.toml",
    )

    dashboard.serve("postgresql://reader:test@127.0.0.1/test", port=0)

    assert calls == ["http://127.0.0.1:8432/mcp", "close"]
    output = capsys.readouterr().out
    assert "Codex MCP config:" in output
    assert "Kaggriculture MCP: http://127.0.0.1:8432/mcp" in output

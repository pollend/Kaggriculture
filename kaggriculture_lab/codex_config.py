"""Project-scoped Codex MCP configuration managed by the dashboard."""

from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path

SERVER_NAME = "kaggriculture_progress"
MANAGED_START = "# BEGIN KAGGRICULTURE DASHBOARD MCP (managed)"
MANAGED_END = "# END KAGGRICULTURE DASHBOARD MCP (managed)"


class CodexConfigConflict(ValueError):
    """The target config contains an unmanaged table owned by the user."""


def _managed_block(servers: dict[str, str]) -> str:
    sections = [MANAGED_START]
    for name, url in sorted(servers.items()):
        sections.extend(
            (
                f"[mcp_servers.{name}]",
                f'url = "{url}"',
                "enabled = true",
                "required = false",
                "tool_timeout_sec = 30",
                'default_tools_approval_mode = "auto"',
                "",
            )
        )
    sections.append(MANAGED_END)
    return "\n".join(sections) + "\n"


def _managed_servers(block: str) -> dict[str, str]:
    document = tomllib.loads(block)
    servers = document.get("mcp_servers", {})
    return {
        name: values["url"]
        for name, values in servers.items()
        if isinstance(values, dict) and isinstance(values.get("url"), str)
    }


@contextmanager
def _config_lock(destination: Path) -> Iterator[None]:
    """Serialize dashboard startups that update the same project config."""

    lock_path = destination.with_name(f".{destination.name}.kaggriculture.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        flock(lock_file.fileno(), LOCK_EX)
        try:
            yield
        finally:
            flock(lock_file.fileno(), LOCK_UN)


def sync_codex_mcp_config(
    url: str,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Atomically synchronize the dashboard URL into project Codex config.

    Only this module's marked block is replaced. An unmanaged table with the
    same server name is treated as user-owned and is never rewritten.
    """

    if not url.startswith(("http://", "https://")):
        raise ValueError("MCP URL must use http or https")
    root = Path.cwd() if project_root is None else Path(project_root)
    destination = root / ".codex" / "config.toml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _config_lock(destination):
        existing = destination.read_text(encoding="utf-8") if destination.exists() else ""
        start = existing.find(MANAGED_START)
        end = existing.find(MANAGED_END)
        if (start < 0) != (end < 0) or (start >= 0 and end < start):
            raise CodexConfigConflict(
                "Codex config contains an incomplete Kaggriculture managed block"
            )
        if existing.count(MANAGED_START) > 1 or existing.count(MANAGED_END) > 1:
            raise CodexConfigConflict(
                "Codex config contains multiple Kaggriculture managed blocks"
            )

        if start >= 0:
            _managed_servers(existing[start : end + len(MANAGED_END)])
        elif SERVER_NAME in tomllib.loads(existing or "").get("mcp_servers", {}):
            table = f"[mcp_servers.{SERVER_NAME}]"
            raise CodexConfigConflict(
                f"Codex config already contains unmanaged {table}; leaving it unchanged"
            )
        # PostgreSQL makes every league available through one shared service.
        # Replacing the complete managed set also migrates old per-run entries.
        block = _managed_block({SERVER_NAME: url})

        if start >= 0:
            end += len(MANAGED_END)
            if end < len(existing) and existing[end] == "\n":
                end += 1
            updated = existing[:start] + block + existing[end:]
        else:
            separator = "" if not existing else "\n" if existing.endswith("\n") else "\n\n"
            updated = existing + separator + block

        # Validate the complete merged document before replacing the user's file.
        tomllib.loads(updated)
        previous_mode = destination.stat().st_mode & 0o777 if destination.exists() else 0o664
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="config.toml.",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary.write(updated)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.chmod(temporary_name, previous_mode)
            os.replace(temporary_name, destination)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
    return destination


def dashboard_mcp_url(host: str, port: int) -> str:
    """Return a local-client URL for a bound dashboard address."""

    client_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    if ":" in client_host and not client_host.startswith("["):
        client_host = f"[{client_host}]"
    return f"http://{client_host}:{port}/mcp"

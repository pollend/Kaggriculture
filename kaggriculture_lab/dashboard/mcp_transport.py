"""The Streamable HTTP transport in front of the MCP backend.

Kept apart from the HTML handler so this is the only dashboard module that knows
`progress_mcp` exists, and so protocol negotiation reads as one thing.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

from kaggriculture_lab.progress_mcp import PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS

MAX_BODY_BYTES = 1_048_576


def send_error(handler: Any, status: HTTPStatus, code: int, message: str, data: Any = None) -> None:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    handler._send(
        status,
        json.dumps({"jsonrpc": "2.0", "id": None, "error": error}, sort_keys=True).encode(),
        "application/json; charset=utf-8",
        head=False,
        headers={"MCP-Protocol-Version": PROTOCOL_VERSION},
    )


def handle_mcp(handler: Any) -> None:
    origin = handler.headers.get("Origin")
    if origin:
        parsed_origin = urlsplit(origin)
        request_host = handler.headers.get("Host", "")
        if (
            parsed_origin.scheme not in {"http", "https"}
            or parsed_origin.netloc.casefold() != request_host.casefold()
        ):
            send_error(handler, HTTPStatus.FORBIDDEN, -32000, "Invalid Origin")
            return
    protocol_version = handler.headers.get("MCP-Protocol-Version")
    if protocol_version and protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
        send_error(
            handler,
            HTTPStatus.BAD_REQUEST,
            -32602,
            "Unsupported MCP protocol version",
            {"supported": list(SUPPORTED_PROTOCOL_VERSIONS), "requested": protocol_version},
        )
        return
    content_type = handler.headers.get("Content-Type", "").partition(";")[0].strip().casefold()
    if content_type != "application/json":
        send_error(
            handler,
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            -32600,
            "Content-Type must be application/json",
        )
        return
    try:
        length = int(handler.headers.get("Content-Length", ""))
    except ValueError:
        send_error(handler, HTTPStatus.LENGTH_REQUIRED, -32600, "Content-Length is required")
        return
    if length < 1:
        send_error(handler, HTTPStatus.BAD_REQUEST, -32700, "Request body is empty")
        return
    if length > MAX_BODY_BYTES:
        send_error(
            handler, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, -32600, "Request body exceeds 1 MiB"
        )
        return
    try:
        message = json.loads(handler.rfile.read(length))
    except (UnicodeDecodeError, json.JSONDecodeError):
        send_error(handler, HTTPStatus.BAD_REQUEST, -32700, "Parse error")
        return
    response = handler.mcp_backend.handle(message)
    negotiated = None
    if response and isinstance(response.get("result"), dict):
        negotiated = response["result"].get("protocolVersion")
    headers = {"MCP-Protocol-Version": protocol_version or negotiated or PROTOCOL_VERSION}
    if response is None:
        handler._send(
            HTTPStatus.ACCEPTED, b"", "application/json; charset=utf-8", head=False, headers=headers
        )
    else:
        handler._send(
            HTTPStatus.OK,
            json.dumps(response, sort_keys=True).encode(),
            "application/json; charset=utf-8",
            head=False,
            headers=headers,
        )

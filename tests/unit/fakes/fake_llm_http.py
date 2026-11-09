"""Scripted HTTP stand-in for the Anthropic Messages and OpenAI Chat Completions APIs.

A network-boundary double (CLAUDE.md §1.11): the real ``anthropic`` / ``openai``
SDKs and the real ``AnthropicToolUseClient`` / ``OpenAICompatibleToolUseClient``
talk to it over loopback HTTP, so SDK error classes, status codes and response
parsing are all exercised for real. Each request pops the next scripted
``(status, body)`` reply and is recorded for assertions.

Production code never imports this module.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

__all__ = [
    "FakeLLMServer",
    "anthropic_error",
    "anthropic_text",
    "anthropic_tool_use",
    "openai_error",
    "openai_text",
    "openai_tool_call",
]

Reply = tuple[int, dict[str, Any]]


def anthropic_tool_use(name: str, arguments: dict[str, Any]) -> Reply:
    """A 200 Messages reply carrying one ``tool_use`` block (after an empty thinking block)."""
    return 200, {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": "fake",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_fake", "name": name, "input": arguments},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def anthropic_text(text: str, *, stop_reason: str = "end_turn") -> Reply:
    """A 200 Messages reply with prose only (no tool call)."""
    return 200, {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": "fake",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def anthropic_error(message: str, *, status: int = 400) -> Reply:
    """An Anthropic API error body."""
    return status, {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }


def openai_tool_call(name: str, arguments: dict[str, Any]) -> Reply:
    """A 200 Chat Completions reply carrying one tool call."""
    return 200, {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fake",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def openai_text(text: str, *, finish_reason: str = "stop") -> Reply:
    """A 200 Chat Completions reply with prose only (no tool call)."""
    return 200, {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def openai_error(message: str, *, status: int = 400, param: str | None = None) -> Reply:
    """An OpenAI API error body."""
    return status, {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": None,
        }
    }


class FakeLLMServer:
    """Loopback HTTP server replaying scripted LLM replies.

    Use as a context manager. ``base_url`` serves both SDKs: Anthropic posts to
    ``/v1/messages`` from ``base_url``; OpenAI posts to ``/chat/completions``
    under ``openai_base_url`` (``base_url + "/v1"``).

    Args:
        replies: ``(status, json_body)`` pairs, consumed one per request.

    Attributes:
        requests: Every request body received, in order.
    """

    def __init__(self, replies: list[Reply]) -> None:
        """Bind an ephemeral loopback port; serving starts on ``__enter__``."""
        self._replies: deque[Reply] = deque(replies)
        self.requests: list[dict[str, Any]] = []
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                server.requests.append(json.loads(self.rfile.read(length) or b"{}"))
                if server._replies:
                    status, body = server._replies.popleft()
                else:
                    status, body = 500, {"error": {"message": "fake LLM: no scripted reply left"}}
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:<port>``."""
        host, port = self._httpd.server_address[:2]
        return f"http://{host!s}:{port}"

    @property
    def openai_base_url(self) -> str:
        """``base_url`` with the ``/v1`` prefix the OpenAI SDK expects."""
        return f"{self.base_url}/v1"

    def __enter__(self) -> FakeLLMServer:
        """Start serving."""
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop serving and release the port."""
        self._httpd.shutdown()
        self._httpd.server_close()

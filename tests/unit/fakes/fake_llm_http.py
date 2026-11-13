"""Scripted HTTP stand-in for the Anthropic Messages and OpenAI Chat Completions / Responses APIs.

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
    "responses_function_call",
    "responses_incomplete",
    "responses_refusal",
    "responses_text",
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


def _response(output: list[dict[str, Any]], **extra: Any) -> Reply:
    """A 200 Responses-API body around ``output`` items."""
    body: dict[str, Any] = {
        "id": "resp_fake",
        "object": "response",
        "created_at": 0,
        "model": "fake",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "required",
        "tools": [],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    body.update(extra)
    return 200, body


def _message(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_fake",
        "role": "assistant",
        "status": "completed",
        "content": content,
    }


def responses_function_call(name: str, arguments: dict[str, Any]) -> Reply:
    """A 200 Responses reply carrying one ``function_call`` (after a reasoning item)."""
    return _response(
        [
            {"type": "reasoning", "id": "rs_fake", "summary": []},
            {
                "type": "function_call",
                "id": "fc_fake",
                "call_id": "call_fake",
                "name": name,
                "arguments": json.dumps(arguments),
                "status": "completed",
            },
        ]
    )


def responses_text(text: str) -> Reply:
    """A 200 Responses reply with prose only (no function call)."""
    return _response([_message([{"type": "output_text", "text": text, "annotations": []}])])


def responses_refusal(text: str) -> Reply:
    """A 200 Responses reply whose message is a refusal."""
    return _response([_message([{"type": "refusal", "refusal": text}])])


def responses_incomplete(reason: str) -> Reply:
    """A 200 Responses reply cut off (``max_output_tokens`` / ``content_filter``)."""
    return _response(
        [{"type": "reasoning", "id": "rs_fake", "summary": []}],
        status="incomplete",
        incomplete_details={"reason": reason},
    )


class FakeLLMServer:
    """Loopback HTTP server replaying scripted LLM replies.

    Use as a context manager. ``base_url`` serves both SDKs: Anthropic posts to
    ``/v1/messages`` from ``base_url``; OpenAI posts to ``/chat/completions`` or
    ``/responses`` under ``openai_base_url`` (``base_url + "/v1"``).

    Args:
        replies: ``(status, json_body)`` pairs, consumed one per POST.
        get_bodies: JSON bodies served (200) on GET by path, e.g. Ollama's
            ``/api/ps``; any other GET is a 404.

    Attributes:
        requests: Every request body received, in order.
        paths: The request path of every request, in order.
    """

    def __init__(
        self, replies: list[Reply], *, get_bodies: dict[str, object] | None = None
    ) -> None:
        """Bind an ephemeral loopback port; serving starts on ``__enter__``."""
        self._replies: deque[Reply] = deque(replies)
        self._get_bodies = dict(get_bodies or {})
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                server.requests.append(json.loads(self.rfile.read(length) or b"{}"))
                server.paths.append(self.path)
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

            def do_GET(self) -> None:
                found = self.path in server._get_bodies
                body = server._get_bodies.get(self.path, {"error": "not found"})
                payload = json.dumps(body).encode()
                self.send_response(200 if found else 404)
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

"""TDD tests for ``describe_image`` on both tool-use clients.

``describe_image`` is the single-shot "ask the LLM about a camera frame"
primitive (image_jpeg + question → text) used by the VLM-adjudicated
completion gate (ADR-0073/C).

§1.11 rule: the ONLY doubles are the ``anthropic`` / ``openai`` SDK objects
at the network boundary — tiny ``SimpleNamespace`` fakes returning canned
responses.  Real JPEG bytes are produced via ``PIL.Image``.

Run with::

    WT=/home/allopart/workspace/openral/.claude/worktrees/vlm-completion
    PYTHONPATH=$WT/python/reasoner/src:$WT/python/core/src \\
      /home/allopart/workspace/openral/.venv/bin/python \\
      -m pytest tests/unit/test_tool_use_describe_image.py -v
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
from openral_reasoner.tool_use import AnthropicToolUseClient, OpenAICompatibleToolUseClient

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CANNED_ANSWER = "Yes, the task is complete."
_CANNED_REASONING = "I reasoned that the object is in the target position."


def _make_jpeg_bytes() -> bytes:
    """Return real 1×1 JPEG bytes via Pillow (real pixels, not b'fake')."""
    try:
        from PIL import (
            Image,  # type: ignore[import-untyped]  # reason: optional dep; skip if absent
        )
    except ImportError:
        pytest.skip("Pillow not installed — cannot produce real JPEG fixture")
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(128, 0, 64)).save(buf, "JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fake Anthropic SDK (network-boundary double, §1.11)
# ---------------------------------------------------------------------------


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answer: str,
    reasoning: str = "",
) -> list[dict[str, object]]:
    """Replace ``anthropic.Anthropic`` with a recorded-response double.

    Returns a ``captured`` list; each element is the kwargs dict passed to
    ``messages.create`` so the test can inspect the request payload.
    """
    captured: list[dict[str, object]] = []

    def _create(**kwargs: object) -> SimpleNamespace:
        captured.append(dict(kwargs))
        if answer:
            content_block = SimpleNamespace(type="text", text=answer)
            return SimpleNamespace(content=[content_block], reasoning="")
        # Reasoning-model path: empty content, answer lives in ``reasoning``.
        return SimpleNamespace(content=[], reasoning=reasoning)

    class _FakeAnthropic:
        def __init__(self, **_kwargs: object) -> None:
            self.messages = SimpleNamespace(create=_create)

    import anthropic  # reason: network-boundary double per §1.11

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    return captured


# ---------------------------------------------------------------------------
# Fake OpenAI SDK (network-boundary double, §1.11)
# ---------------------------------------------------------------------------


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answer: str,
    reasoning: str = "",
) -> list[dict[str, object]]:
    """Replace ``openai.OpenAI`` with a recorded-response double.

    Returns a ``captured`` list of kwargs dicts passed to
    ``chat.completions.create``.
    """
    captured: list[dict[str, object]] = []

    def _create(**kwargs: object) -> SimpleNamespace:
        captured.append(dict(kwargs))
        message = SimpleNamespace(content=answer, reasoning=reasoning)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    import openai  # reason: network-boundary double per §1.11

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return captured


# ---------------------------------------------------------------------------
# AnthropicToolUseClient.describe_image
# ---------------------------------------------------------------------------


def test_anthropic_describe_image_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returned text equals the canned answer from the provider."""
    jpeg = _make_jpeg_bytes()
    _install_fake_anthropic(monkeypatch, answer=_CANNED_ANSWER)

    client = AnthropicToolUseClient(model_id="claude-haiku-4-5", api_key="sk-ant-test")
    result = client.describe_image(image_jpeg=jpeg, question="Is the task done?")

    assert result == _CANNED_ANSWER


def test_anthropic_describe_image_sends_base64_image_and_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request carries a base64 image block + the question text block."""
    jpeg = _make_jpeg_bytes()
    captured = _install_fake_anthropic(monkeypatch, answer=_CANNED_ANSWER)

    client = AnthropicToolUseClient(model_id="claude-haiku-4-5", api_key="sk-ant-test")
    client.describe_image(image_jpeg=jpeg, question="Is the task done?")

    assert len(captured) == 1
    messages = captured[0]["messages"]
    assert isinstance(messages, list) and len(messages) == 1
    user_content = messages[0]["content"]
    assert isinstance(user_content, list)

    # Image block
    image_blocks = [b for b in user_content if isinstance(b, dict) and b.get("type") == "image"]
    assert len(image_blocks) == 1, f"Expected one image block, got: {user_content}"
    src = image_blocks[0]["source"]
    assert isinstance(src, dict)
    assert src["type"] == "base64"
    assert src["media_type"] == "image/jpeg"
    expected_b64 = base64.b64encode(jpeg).decode()
    assert src["data"] == expected_b64

    # Text / question block
    text_blocks = [b for b in user_content if isinstance(b, dict) and b.get("type") == "text"]
    assert len(text_blocks) == 1, f"Expected one text block, got: {user_content}"
    assert text_blocks[0]["text"] == "Is the task done?"


def test_anthropic_describe_image_reasoning_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``content`` is empty, the ``reasoning`` field is returned instead."""
    jpeg = _make_jpeg_bytes()
    _install_fake_anthropic(monkeypatch, answer="", reasoning=_CANNED_REASONING)

    client = AnthropicToolUseClient(model_id="claude-haiku-4-5", api_key="sk-ant-test")
    result = client.describe_image(image_jpeg=jpeg, question="Done?")

    assert result == _CANNED_REASONING


# ---------------------------------------------------------------------------
# OpenAICompatibleToolUseClient.describe_image
# ---------------------------------------------------------------------------


def test_openai_describe_image_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returned text equals the canned answer from the provider."""
    jpeg = _make_jpeg_bytes()
    _install_fake_openai(monkeypatch, answer=_CANNED_ANSWER)

    client = OpenAICompatibleToolUseClient(model_id="gpt-4o-mini", api_key="sk-test")
    result = client.describe_image(image_jpeg=jpeg, question="Is the task done?")

    assert result == _CANNED_ANSWER


def test_openai_describe_image_sends_image_url_and_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request carries an ``image_url`` block with data-URI + the question."""
    jpeg = _make_jpeg_bytes()
    captured = _install_fake_openai(monkeypatch, answer=_CANNED_ANSWER)

    client = OpenAICompatibleToolUseClient(model_id="gpt-4o-mini", api_key="sk-test")
    client.describe_image(image_jpeg=jpeg, question="Is the task done?")

    assert len(captured) == 1
    messages = captured[0]["messages"]
    assert isinstance(messages, list)
    user_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    assert user_msg is not None, f"No user message in {messages}"
    content = user_msg["content"]
    assert isinstance(content, list)

    # image_url block
    image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image_url"]
    assert len(image_blocks) == 1, f"Expected one image_url block, got: {content}"
    expected_b64 = base64.b64encode(jpeg).decode()
    assert image_blocks[0]["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"

    # text block
    text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
    assert len(text_blocks) == 1, f"Expected one text block, got: {content}"
    assert text_blocks[0]["text"] == "Is the task done?"


def test_openai_describe_image_reasoning_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``message.content`` is empty, ``message.reasoning`` is returned instead.

    Observed with ``z-ai/glm-5.2`` on OpenRouter where reasoning models surface
    their answer in a ``reasoning`` field and leave ``content`` empty.
    """
    jpeg = _make_jpeg_bytes()
    _install_fake_openai(monkeypatch, answer="", reasoning=_CANNED_REASONING)

    client = OpenAICompatibleToolUseClient(
        model_id="z-ai/glm-5.2",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
    )
    result = client.describe_image(image_jpeg=jpeg, question="Done?")

    assert result == _CANNED_REASONING

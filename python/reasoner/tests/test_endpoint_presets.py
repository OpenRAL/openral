"""Named ``OPENRAL_REASONER_ENDPOINT`` values (ADR-0088 escape hatch).

The point of the presets is that everything the retired
``OPENRAL_REASONER_LLM_PROVIDER`` shim could express is expressible
model-first. That is what made the shim deletable in 0.3.0 rather than
load-bearing, so these tests guard the replacement contract.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from openral_core.exceptions import ROSConfigError
from openral_reasoner.tool_use import (
    ANTHROPIC_BASE_URL,
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    OpenAICompatibleToolUseClient,
    build_tool_use_client_from_env,
)

pytest.importorskip("openai", reason="OpenAI SDK not installed")

_MODEL = "qwen3:4b"


@pytest.fixture(autouse=True)
def _clear_reasoner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "OPENRAL_REASONER_MODEL",
        "OPENRAL_REASONER_ENDPOINT",
        "OPENRAL_REASONER_API_KEY",
        "OPENRAL_REASONER_DIALECT",
        "OPENRAL_REASONER_MAX_TOKENS",
        "OPENRAL_REASONER_TIMEOUT_S",
        "OPENRAL_REASONER_LLM_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)


def test_local_preset_needs_no_dialect_or_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ENDPOINT=ollama`` alone resolves URL + dialect + cold-start timeout."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", _MODEL)
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "ollama")

    client = build_tool_use_client_from_env()

    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == _MODEL
    assert client._base_url == OLLAMA_BASE_URL
    # 60 s, not the 10 s cloud default: a cold Ollama model materialises slowly.
    assert client._timeout_s == 60.0


def test_cloud_preset_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "google/gemma-4-26b-a4b-it:free")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "openrouter")

    with pytest.raises(ROSConfigError, match="OPENRAL_REASONER_API_KEY"):
        build_tool_use_client_from_env()

    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-test")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == OPENROUTER_BASE_URL


def test_huggingface_preset_keeps_auto_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HF router 400s on ``tool_choice="required"`` — the preset carries that."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "huggingface")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "hf-test")

    client = build_tool_use_client_from_env()

    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._tool_choice == "auto"


def test_bare_url_still_demands_a_dialect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing can classify an arbitrary URL, so that spelling keeps the ask."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", _MODEL)
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "http://10.0.0.5:9000/v1")

    with pytest.raises(ROSConfigError, match="OPENRAL_REASONER_DIALECT"):
        build_tool_use_client_from_env()

    monkeypatch.setenv("OPENRAL_REASONER_DIALECT", "openai")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == "http://10.0.0.5:9000/v1"


def test_explicit_dialect_overrides_a_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A preset fronted by a translating proxy stays configurable."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", _MODEL)
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "vllm")
    monkeypatch.setenv("OPENRAL_REASONER_DIALECT", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-test")

    client = build_tool_use_client_from_env()

    assert type(client).__name__ == "AnthropicToolUseClient"


def test_no_choices_reports_the_provider_error() -> None:
    """A gateway 200 with ``choices: null`` must name the upstream failure.

    OpenRouter answers HTTP 200 with ``choices: null`` when the backing
    provider rate-limits or 5xxs; ``list(None)`` used to surface as
    ``TypeError: 'NoneType' object is not iterable``, which tells an operator
    nothing about a transient provider outage.
    """
    from openral_core.exceptions import ROSPlanningError
    from openral_reasoner.tool_use import _openai_choices

    class _GatewayError:
        choices: ClassVar[None] = None
        error: ClassVar[dict[str, object]] = {"code": 429, "message": "rate-limited upstream"}

    with pytest.raises(ROSPlanningError, match="rate-limited upstream"):
        _openai_choices(_GatewayError())

    class _Ok:
        choices: ClassVar[list[str]] = ["c0"]

    assert _openai_choices(_Ok()) == ["c0"]


def test_anthropic_preset_is_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ENDPOINT=anthropic` must build a client, not reject its own name.

    The preset's `url` was `None` ("SDK default host"), which collided with the
    `endpoint is None` sentinel meaning "not configured" — so the only preset
    with no explicit URL was the only one that could never be used.
    """
    pytest.importorskip("anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-test")

    client = build_tool_use_client_from_env()

    assert type(client).__name__ == "AnthropicToolUseClient"
    assert client._base_url == ANTHROPIC_BASE_URL


def test_curated_model_takes_the_whole_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A named endpoint contributes tool_choice + timeout too, not only its URL.

    Taking `preset.url` alone built `gpt-5.5` against the HF router with
    `tool_choice="required"` — the 400 the preset exists to avoid.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "huggingface")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "hf-test")

    client = build_tool_use_client_from_env()

    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == "https://router.huggingface.co/v1"
    assert client._tool_choice == "auto"  # endpoint property, not the registry's "required"
    assert client._timeout_s == 60.0  # HF router cold-starts serverless models


def test_curated_model_rejects_a_dialect_clash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A named endpoint cannot re-dialect a curated model.

    This previously built an `AnthropicToolUseClient` pointed at Ollama's
    OpenAI-only endpoint: it configured cleanly and failed on every tick.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "ollama")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-test")

    with pytest.raises(ROSConfigError, match="cannot re-dialect"):
        build_tool_use_client_from_env()


def test_bare_url_still_waives_auth_for_a_curated_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unchanged: a URL means the operator owns the endpoint, so no key is forced.

    A named endpoint states its own auth posture, so that path still demands one.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "https://my-proxy.internal/v1")

    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == "https://my-proxy.internal/v1"

    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "openrouter")
    monkeypatch.delenv("OPENRAL_REASONER_API_KEY", raising=False)
    with pytest.raises(ROSConfigError, match="OPENRAL_REASONER_API_KEY"):
        build_tool_use_client_from_env()

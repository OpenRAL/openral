"""Named ``OPENRAL_REASONER_ENDPOINT`` values (ADR-0088 escape hatch).

The point of the presets is that everything the deprecated
``OPENRAL_REASONER_LLM_PROVIDER`` shim could express is expressible
model-first — otherwise the shim is load-bearing, not deprecated.
"""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSConfigError
from openral_reasoner.tool_use import (
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

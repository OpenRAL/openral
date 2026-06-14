"""Unit tests for :func:`openral_reasoner.build_tool_use_client_from_env`.

Drives the env-driven factory through every branch — anthropic /
openai-compatible / openrouter / ollama / unknown / missing-required —
with ``monkeypatch.setenv`` only. No mocks of openral types per
CLAUDE.md §1.11; we inspect the constructed object's attributes directly.
"""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSConfigError
from openral_reasoner.tool_use import (
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    AnthropicToolUseClient,
    OpenAICompatibleToolUseClient,
    build_tool_use_client_from_env,
)

_ENV_VARS = (
    "OPENRAL_REASONER_LLM_PROVIDER",
    "OPENRAL_REASONER_LLM_MODEL",
    "OPENRAL_REASONER_LLM_API_KEY",
    "OPENRAL_REASONER_LLM_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip OPENRAL_REASONER_LLM_* before every test so a stray
    developer env doesn't shadow the case under test."""
    for key in _ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def test_provider_unset_raises_with_message() -> None:
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    msg = str(excinfo.value)
    assert "OPENRAL_REASONER_LLM_PROVIDER" in msg
    # The error must list every accepted value so the user sees the menu.
    assert "anthropic" in msg
    assert "openai-compatible" in msg
    assert "openrouter" in msg
    assert "ollama" in msg


def test_provider_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "groq-cloud")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "some-model")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    msg = str(excinfo.value)
    assert "groq-cloud" in msg
    assert "anthropic" in msg


def test_model_unset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-ant-x")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_LLM_MODEL" in str(excinfo.value)


def test_anthropic_builds_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-ant-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, AnthropicToolUseClient)
    assert client.model_id == "claude-haiku-4-5"


def test_anthropic_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "claude-haiku-4-5")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_LLM_API_KEY" in str(excinfo.value)
    assert "anthropic" in str(excinfo.value)


def test_openai_compatible_uses_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_BASE_URL", "http://localhost:11434/v1")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == "qwen3:8b"
    # _base_url is a private attribute but exposed in tests is fine —
    # the round-trip is what matters and the client has no public
    # accessor (matches the openai SDK pattern).
    assert client._base_url == "http://localhost:11434/v1"
    # Local Ollama / vLLM endpoints commonly don't enforce auth, so
    # the api_key is allowed to be None.
    assert client._api_key is None


def test_openai_compatible_no_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare openai-compatible defaults the base_url to None — the openai
    SDK then points at api.openai.com (the documented behaviour)."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-openai")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url is None
    assert client._api_key == "sk-openai"


def test_openrouter_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "deepseek/deepseek-chat-v3:free")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-or-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == "deepseek/deepseek-chat-v3:free"
    assert client._base_url == OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_openrouter_explicit_base_url_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit BASE_URL overrides the OpenRouter default (proxy / staging)."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "anthropic/claude-haiku-4.5")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-or-secret")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_BASE_URL", "https://openrouter-proxy.internal/v1")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == "https://openrouter-proxy.internal/v1"


def test_openrouter_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "deepseek/deepseek-chat-v3:free")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_LLM_API_KEY" in str(excinfo.value)
    assert "openrouter" in str(excinfo.value)


def test_ollama_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ollama` with no BASE_URL pins to the local default and needs no key."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "qwen3:0.6b")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == OLLAMA_BASE_URL == "http://localhost:11434/v1"


def test_ollama_explicit_base_url_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user-provided BASE_URL still overrides the default — e.g. a hosted Ollama gateway."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "tenant-token-xyz")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_BASE_URL", "https://ollama-gateway.internal/v1")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == "https://ollama-gateway.internal/v1"

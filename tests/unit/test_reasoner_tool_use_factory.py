"""Unit tests for :func:`openral_reasoner.build_tool_use_client_from_env`.

Drives the env-driven factory through every branch — anthropic /
openai-compatible / openrouter / ollama / gemini / xai / deepseek /
unknown / missing-required — with ``monkeypatch.setenv`` only. No mocks
of openral types per CLAUDE.md §1.11; we inspect the constructed
object's attributes directly.
"""

from __future__ import annotations

import pytest
from openral_core import REASONER_MODELS
from openral_core.exceptions import ROSConfigError
from openral_reasoner.cosmos3 import Cosmos3ToolUseClient
from openral_reasoner.tool_use import (
    DEEPSEEK_BASE_URL,
    GEMINI_BASE_URL,
    HUGGINGFACE_BASE_URL,
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    VLLM_BASE_URL,
    XAI_BASE_URL,
    AnthropicToolUseClient,
    OpenAICompatibleToolUseClient,
    build_tool_use_client_from_env,
)

_ENV_VARS = (
    "OPENRAL_REASONER_MODEL",
    "OPENRAL_REASONER_ENDPOINT",
    "OPENRAL_REASONER_API_KEY",
    "OPENRAL_REASONER_DIALECT",
    "OPENRAL_REASONER_MAX_TOKENS",
    "OPENRAL_REASONER_TIMEOUT_S",
    "OPENRAL_REASONER_LLM_PROVIDER",
    "OPENRAL_REASONER_LLM_MODEL",
    "OPENRAL_REASONER_LLM_API_KEY",
    "OPENRAL_REASONER_LLM_BASE_URL",
    "OPENRAL_REASONER_LLM_MAX_TOKENS",
    "OPENRAL_REASONER_LLM_TIMEOUT_S",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip model-first + legacy reasoner env before every test."""
    for key in _ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def test_model_unset_raises_with_message() -> None:
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    msg = str(excinfo.value)
    assert "OPENRAL_REASONER_MODEL" in msg
    assert "claude-opus-4-8" in msg
    assert "gpt-5.5" in msg
    assert "gpt-5.6" in msg
    assert "cosmos3-edge" in msg
    assert "OPENRAL_REASONER_LLM_PROVIDER" in msg


def test_curated_anthropic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-ant-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, AnthropicToolUseClient)
    assert client.model_id == "claude-opus-4-8"


def test_curated_anthropic_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "https://anthropic-proxy.internal")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "proxy-key")
    monkeypatch.setenv("OPENRAL_REASONER_MAX_TOKENS", "4096")
    client = build_tool_use_client_from_env()
    assert isinstance(client, AnthropicToolUseClient)
    assert client._base_url == "https://anthropic-proxy.internal"
    assert client._max_tokens == 4096


@pytest.mark.parametrize("model_key", ["gpt-5.5", "gpt-5.6"])
def test_curated_openrouter_models(monkeypatch: pytest.MonkeyPatch, model_key: str) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", model_key)
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-or-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == REASONER_MODELS[model_key].served_model_id
    assert client._base_url == OPENROUTER_BASE_URL
    assert client._max_tokens == 16384


def test_curated_cosmos_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client.model_id == "nvidia/Cosmos3-Edge"


def test_uncurated_model_requires_explicit_endpoint_and_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "qwen3:8b")
    with pytest.raises(ROSConfigError, match="OPENRAL_REASONER_ENDPOINT"):
        build_tool_use_client_from_env()

    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENRAL_REASONER_DIALECT", "openai")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == "qwen3:8b"
    assert client._base_url == "http://localhost:11434/v1"


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


def test_max_tokens_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No OPENRAL_REASONER_LLM_MAX_TOKENS → client sends no cap (endpoint default)."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "openai/gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-or-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._max_tokens is None


def test_max_tokens_env_caps_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENRAL_REASONER_LLM_MAX_TOKENS bounds a metered gateway's reservation (HTTP 402)."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "openai/gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-or-secret")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MAX_TOKENS", "8000")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._max_tokens == 8000


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


def test_vllm_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """`vllm` with no BASE_URL pins to the local vLLM default and needs no key."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "vllm")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "qwen2.5-7b-instruct")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == "qwen2.5-7b-instruct"
    assert client._base_url == VLLM_BASE_URL == "http://localhost:8000/v1"
    # Self-hosted vLLM enforces no auth by default.
    assert client._api_key is None


def test_vllm_explicit_base_url_and_key_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit BASE_URL / API_KEY override the vLLM defaults — e.g. a
    remote `vllm serve --api-key` deployment on a non-default port."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "vllm")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "served-key")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_BASE_URL", "http://gpu-box.internal:8001/v1")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == "http://gpu-box.internal:8001/v1"
    assert client._api_key == "served-key"


# ── Named cloud presets (gemini / xai / deepseek) ──────────────────────────────
# Thin auth-required convenience wrappers over OpenAICompatibleToolUseClient
# that pre-fill the vendor's own OpenAI-compatible base URL (issue #74).


@pytest.mark.parametrize(
    ("provider", "model", "expected_base_url"),
    [
        ("gemini", "gemini-2.5-flash", GEMINI_BASE_URL),
        ("xai", "grok-4", XAI_BASE_URL),
        ("deepseek", "deepseek-chat", DEEPSEEK_BASE_URL),
    ],
)
def test_named_preset_default_base_url(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    expected_base_url: str,
) -> None:
    """Each named preset pins the vendor base URL and forwards model + key."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", model)
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", f"sk-{provider}-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == model
    assert client._base_url == expected_base_url
    assert client._api_key == f"sk-{provider}-secret"


@pytest.mark.parametrize("provider", ["gemini", "xai", "deepseek"])
def test_named_preset_explicit_base_url_wins(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """An explicit BASE_URL overrides a named preset's default (proxy / staging)."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "some-model")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_BASE_URL", "https://llm-proxy.internal/v1")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == "https://llm-proxy.internal/v1"


@pytest.mark.parametrize("provider", ["gemini", "xai", "deepseek", "huggingface"])
def test_named_preset_without_key_raises(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    """The named presets all enforce auth, like openrouter."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "some-model")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_LLM_API_KEY" in str(excinfo.value)
    assert provider in str(excinfo.value)


# ── huggingface preset (HF router; tool_choice=auto) ───────────────────────────


def test_huggingface_default_base_url_and_auto_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`huggingface` pins the HF router base URL and uses tool_choice=auto.

    The HF router rejects tool_choice=required (400 INVALID_TOOL_CHOICE), so the
    preset must select "auto" — unlike the other cloud presets which force "required".
    """
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "hf-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == "Qwen/Qwen3-8B"
    assert client._base_url == HUGGINGFACE_BASE_URL == "https://router.huggingface.co/v1"
    assert client._tool_choice == "auto"


def test_other_cloud_presets_force_required_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openrouter (and the other vendor presets) keep tool_choice=required."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "deepseek/deepseek-chat-v3:free")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-or-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._tool_choice == "required"

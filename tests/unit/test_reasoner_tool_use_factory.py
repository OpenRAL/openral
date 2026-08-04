"""Unit tests for :func:`openral_reasoner.build_tool_use_client_from_env`.

Drives the model-first env factory (ADR-0088) through every branch — curated
registry models, the named endpoints (`anthropic` / `openrouter` / `gemini` /
`xai` / `deepseek` / `huggingface` / `ollama` / `vllm`), a bare URL plus an
explicit dialect, and each missing-required case — with ``monkeypatch.setenv``
only. No mocks of openral types per CLAUDE.md §1.11; we inspect the constructed
object's attributes directly.
"""

from __future__ import annotations

import pytest
from openral_core import REASONER_MODELS
from openral_core.exceptions import ROSConfigError
from openral_reasoner.cosmos3 import Cosmos3ToolUseClient
from openral_reasoner.tool_use import (
    ANTHROPIC_BASE_URL,
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
    # Retired provider-first spelling (removed in 0.3.0). Cleared so the
    # "it is ignored" tests below assert on a genuinely absent contract
    # rather than on whatever the dev host happens to export.
    "OPENRAL_REASONER_LLM_PROVIDER",
    "OPENRAL_REASONER_LLM_MODEL",
    "OPENRAL_REASONER_LLM_API_KEY",
    "OPENRAL_REASONER_LLM_BASE_URL",
    "OPENRAL_REASONER_LLM_MAX_TOKENS",
    "OPENRAL_REASONER_LLM_TIMEOUT_S",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every reasoner env var before each test."""
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


# ── The retired provider-first contract ───────────────────────────────────────


def test_legacy_provider_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully-populated legacy env selects nothing — the shim is gone.

    Before 0.3.0 this built an Anthropic client. It must now fail exactly as an
    empty environment does, so a deployment still on the old spelling gets the
    migration message instead of silently reasoning with an unintended model.
    """
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-ant-secret")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_MODEL" in str(excinfo.value)


def test_legacy_base_url_does_not_leak_into_model_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retired BASE_URL var must not steer a model-first build."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_BASE_URL", "https://stale-gateway.internal/v1")
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-or-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == OPENROUTER_BASE_URL


# ── Curated registry models ───────────────────────────────────────────────────


def test_curated_anthropic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-ant-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, AnthropicToolUseClient)
    assert client.model_id == "claude-opus-4-8"


def test_curated_anthropic_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-opus-4-8")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_API_KEY" in str(excinfo.value)
    assert "claude-opus-4-8" in str(excinfo.value)


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


def test_curated_openrouter_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "gpt-5.5")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_API_KEY" in str(excinfo.value)


def test_curated_cosmos_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """`cosmos3-edge` replaces the legacy `PROVIDER=cosmos`, autostart included."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client.model_id == "nvidia/Cosmos3-Edge"


def test_curated_model_accepts_named_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A curated model can be re-pointed at a named endpoint, not just a URL."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "vllm")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._base_url == VLLM_BASE_URL


# ── Uncurated models on a bare URL (needs an explicit dialect) ────────────────


def test_uncurated_model_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "qwen3:8b")
    with pytest.raises(ROSConfigError, match="OPENRAL_REASONER_ENDPOINT"):
        build_tool_use_client_from_env()


def test_uncurated_bare_url_requires_dialect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing can classify a bare URL, so that spelling still needs DIALECT."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "qwen3:8b")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "http://gpu-box.internal:8001/v1")
    with pytest.raises(ROSConfigError, match="OPENRAL_REASONER_DIALECT"):
        build_tool_use_client_from_env()


def test_uncurated_bare_url_with_dialect(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ENDPOINT=<url>` + `DIALECT=openai` replaces the legacy `openai-compatible`."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "qwen3:8b")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENRAL_REASONER_DIALECT", "openai")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == "qwen3:8b"
    assert client._base_url == "http://localhost:11434/v1"
    # A self-hosted endpoint commonly enforces no auth, so a key stays optional.
    assert client._api_key is None


def test_uncurated_unknown_endpoint_name_is_not_a_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognised name is treated as a URL, so it still demands a dialect."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "some-model")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "groq-cloud")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    msg = str(excinfo.value)
    assert "OPENRAL_REASONER_DIALECT" in msg
    # The message lists the names that *are* presets, so the fix is discoverable.
    assert "openrouter" in msg
    assert "ollama" in msg


# ── Named endpoints: auth-required cloud vendors ──────────────────────────────


@pytest.mark.parametrize(
    ("endpoint", "model", "expected_base_url"),
    [
        ("openrouter", "deepseek/deepseek-chat-v3:free", OPENROUTER_BASE_URL),
        ("gemini", "gemini-2.5-flash", GEMINI_BASE_URL),
        ("xai", "grok-4", XAI_BASE_URL),
        ("deepseek", "deepseek-chat", DEEPSEEK_BASE_URL),
        ("huggingface", "Qwen/Qwen3-8B", HUGGINGFACE_BASE_URL),
    ],
)
def test_named_cloud_endpoint_pins_base_url(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    model: str,
    expected_base_url: str,
) -> None:
    """Each named endpoint supplies its own URL, dialect and auth posture."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", model)
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", endpoint)
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", f"sk-{endpoint}-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == model
    assert client._base_url == expected_base_url
    assert client._api_key == f"sk-{endpoint}-secret"


@pytest.mark.parametrize("endpoint", ["openrouter", "gemini", "xai", "deepseek", "huggingface"])
def test_named_cloud_endpoint_without_key_raises(
    monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    """The cloud endpoints all enforce auth."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "some-model")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", endpoint)
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_API_KEY" in str(excinfo.value)
    assert endpoint in str(excinfo.value)


def test_named_anthropic_endpoint_builds_anthropic_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ENDPOINT=anthropic` + a raw model id replaces the legacy PROVIDER=anthropic.

    The preset carries the anthropic dialect, so no DIALECT is needed, and its
    URL is Anthropic's own API host spelled out.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-ant-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, AnthropicToolUseClient)
    assert client.model_id == "claude-haiku-4-5"
    assert client._base_url == ANTHROPIC_BASE_URL


def test_named_anthropic_endpoint_without_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "anthropic")
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "OPENRAL_REASONER_API_KEY" in str(excinfo.value)


# ── Named endpoints: local self-hosted daemons (no auth) ──────────────────────


@pytest.mark.parametrize(
    ("endpoint", "model", "expected_base_url"),
    [
        ("ollama", "qwen3:0.6b", OLLAMA_BASE_URL),
        ("vllm", "qwen2.5-7b-instruct", VLLM_BASE_URL),
    ],
)
def test_named_local_endpoint_needs_no_key(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    model: str,
    expected_base_url: str,
) -> None:
    """`ENDPOINT=ollama|vllm` replaces the legacy local presets, key optional."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", model)
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", endpoint)
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == model
    assert client._base_url == expected_base_url
    assert client._api_key is None


def test_named_local_endpoint_accepts_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway-fronted daemon (or `vllm serve --api-key`) still passes one."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "vllm")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "served-key")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._api_key == "served-key"


# ── Preset-carried properties: dialect, tool_choice, cold-start timeout ───────


def test_explicit_dialect_overrides_a_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit dialect wins, for a preset behind a translating proxy."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_DIALECT", "openai")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "proxy-key")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)


def test_huggingface_endpoint_uses_auto_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HF router 400s on tool_choice=required (INVALID_TOOL_CHOICE)."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "huggingface")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "hf-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._tool_choice == "auto"


def test_other_endpoints_force_required_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "deepseek/deepseek-chat-v3:free")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-or-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._tool_choice == "required"


@pytest.mark.parametrize(
    ("endpoint", "expected_timeout_s"),
    [
        # Loopback daemons and the HF router materialise a model on the first
        # call; the warm cloud vendors stay on the tight default.
        ("ollama", 60.0),
        ("vllm", 60.0),
        ("huggingface", 60.0),
        ("openrouter", 10.0),
    ],
)
def test_named_endpoint_carries_cold_start_timeout(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, expected_timeout_s: float
) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "some-model")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", endpoint)
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "a-key")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._timeout_s == expected_timeout_s


def test_timeout_env_overrides_the_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "ollama")
    monkeypatch.setenv("OPENRAL_REASONER_TIMEOUT_S", "5")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._timeout_s == 5.0


# ── Completion-token cap ──────────────────────────────────────────────────────


def test_max_tokens_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cap → the client sends none and the endpoint applies its own."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "openai/gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-or-secret")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._max_tokens is None


def test_max_tokens_env_caps_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cap bounds a metered gateway's up-front reservation (HTTP 402)."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "openai/gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("OPENRAL_REASONER_MAX_TOKENS", "8000")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._max_tokens == 8000

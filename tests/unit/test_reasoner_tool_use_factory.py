"""Unit tests for ``openral_reasoner.build_tool_use_client_from_env``.

Drives the model-first env factory (ADR-0088) through every branch — curated
registry models, the named endpoints (`anthropic` / `openai` / `openrouter` / `gemini` /
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

from tests.unit.fakes.fake_llm_http import FakeLLMServer, openai_tool_call

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
    assert client._max_tokens == REASONER_MODELS["cosmos3-edge"].max_tokens_default == 512


def test_curated_cosmos_cap_yields_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPENRAL_REASONER_MAX_TOKENS`` lifts the 512 cap, e.g. for long decompositions."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    monkeypatch.setenv("OPENRAL_REASONER_MAX_TOKENS", "2048")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client._max_tokens == 2048


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


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("OPENRAL_REASONER_TIMEOUT_S", "10s"),
        ("OPENRAL_REASONER_MAX_TOKENS", "8k"),
    ],
)
def test_malformed_numeric_env_raises_typed_config_error(
    monkeypatch: pytest.MonkeyPatch, env: str, value: str
) -> None:
    """A typo'd numeric env var is a config error, not a ValueError traceback.

    The factory used to call bare float()/int() on the env value, so
    OPENRAL_REASONER_TIMEOUT_S=10s crashed reasoner bring-up untyped and the
    lifecycle/replanning machinery keyed on ROSError never saw it (§5).
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "openai/gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-or-secret")
    monkeypatch.setenv(env, value)
    with pytest.raises(ROSConfigError, match=env):
        build_tool_use_client_from_env()


# ── Curated Ollama models ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "served"), [("qwen3.5-4b", "qwen3.5:4b"), ("qwen3.5-9b", "qwen3.5:9b")]
)
def test_curated_ollama_model_needs_only_the_model(
    monkeypatch: pytest.MonkeyPatch, model: str, served: str
) -> None:
    """A curated Ollama model alone reaches the local Ollama with the `ollama` preset's properties.

    The website and the robot console set only the model for a no-key model, so the
    registry default (`OLLAMA_BASE_URL`) must bring Ollama's 60 s cold start and
    `auto` tool_choice, not the 10 s cloud default.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", model)
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert (client.model_id, client._base_url, client._api_key) == (
        served,
        OLLAMA_BASE_URL,
        None,
    )
    assert (client._timeout_s, client._tool_choice) == (60.0, "auto")


def test_ollama_endpoint_sends_auto_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama ignores tool_choice, so the preset sends `auto` and keeps the prose nudge."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "ollama")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client._tool_choice == "auto"


def test_a_curated_ollama_model_at_an_explicit_url_keeps_the_local_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit URL (a trailing slash, a remote GPU box) is no named preset.

    The 9B's median tick is 18.8 s, so the 10 s cloud default would time out
    every tick; ``byo_local`` gets the local endpoints' 60 s cold start.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "qwen3.5-9b")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "http://localhost:11434/v1/")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert (client._timeout_s, client._tool_choice) == (60.0, "auto")


def test_an_inferred_preset_never_overrides_the_entry_tool_choice() -> None:
    """A registry default that matches a preset's URL brings its timeout, not its tool_choice.

    A local entry on vLLM's default URL that runs ``auto`` keeps ``auto``; only a
    named ``OPENRAL_REASONER_ENDPOINT`` applies the preset's ``required``.
    """
    from openral_reasoner.tool_use import _build_curated_model

    entry = REASONER_MODELS["qwen3.5-4b"].model_copy(update={"default_endpoint": VLLM_BASE_URL})
    client = _build_curated_model(entry)
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert (client._base_url, client._timeout_s, client._tool_choice) == (
        VLLM_BASE_URL,
        60.0,
        "auto",
    )


def test_an_inferred_preset_never_overrides_a_cloud_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpt-5.5's default URL is OpenRouter's, but its own tool_choice stays in charge."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-test")
    monkeypatch.delenv("OPENRAL_REASONER_ENDPOINT", raising=False)
    client = build_tool_use_client_from_env()
    assert client._tool_choice == "required"  # type: ignore[attr-defined]  # reason: client-private


def _ollama_reply(prompt_tokens: int, fingerprint: str | None) -> tuple[int, dict[str, object]]:
    """A ``wait`` tool call as Ollama's OpenAI endpoint sends it, reading ``prompt_tokens``."""
    status, body = openai_tool_call("wait", {"rationale": "ok"})
    body["system_fingerprint"] = fingerprint
    body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 7, "total_tokens": 0}
    return status, body


@pytest.mark.parametrize(
    ("context", "prompt_tokens", "fingerprint", "cut"),
    [
        ("x" * 31_000, 100_000, "fp_ollama", False),  # the whole prompt read
        ("x" * 31_000, 4096, "fp_ollama", True),  # Ollama's default window
        ("x" * 31_000, 4096, None, False),  # not Ollama: others 400 on an over-long prompt
        # Dense non-ASCII text: counted as characters, not as 6-char JSON escapes.
        ("机器人" * 10_000, 15_000, "fp_ollama", False),
        # Repeated text packs ~8 chars/token: read whole when the reported window holds it.
        ("a" * 100_000, 15_000, "fp_ollama", False),
    ],
)
def test_an_ollama_reply_that_read_a_cut_prompt_is_refused(
    context: str, prompt_tokens: int, fingerprint: str | None, cut: bool
) -> None:
    """The real client refuses an Ollama reply that read far fewer tokens than it sent.

    The fake reports the 4096-token default window (``/api/ps``) for the cut cases
    and a 32K one for the rest, as a real Ollama does.
    """
    from openral_core import WaitTool
    from openral_core.exceptions import ROSPlanningError
    from openral_reasoner.palette import ToolPalette

    window = 4096 if cut else 32768
    ps = {"models": [{"name": "qwen3.5:4b", "model": "qwen3.5:4b", "context_length": window}]}
    replies = [_ollama_reply(prompt_tokens, fingerprint)]
    with FakeLLMServer(replies, get_bodies={"/api/ps": ps}) as server:
        client = OpenAICompatibleToolUseClient(
            model_id="qwen3.5:4b", base_url=server.openai_base_url, tool_choice="auto"
        )
        if cut:
            with pytest.raises(ROSPlanningError, match="cut the prompt"):
                client.select_tool(context_text=context, palette=ToolPalette(skills=()))
        else:
            tool = client.select_tool(context_text=context, palette=ToolPalette(skills=()))
            assert isinstance(tool, WaitTool)
    assert len(server.requests) == 1  # a cut prompt is never nudged and re-sent


def test_a_ratio_flagged_reply_is_refused_without_a_reported_window() -> None:
    """No ``/api/ps`` (an old or proxied Ollama): the ratio alone decides."""
    from openral_core.exceptions import ROSPlanningError
    from openral_reasoner.palette import ToolPalette

    with FakeLLMServer([_ollama_reply(4096, "fp_ollama")]) as server:
        client = OpenAICompatibleToolUseClient(
            model_id="qwen3.5:4b", base_url=server.openai_base_url, tool_choice="auto"
        )
        with pytest.raises(ROSPlanningError, match="cut the prompt"):
            client.select_tool(context_text="x" * 31_000, palette=ToolPalette(skills=()))


@pytest.mark.parametrize("ps", [[], {"models": [None]}, {"models": "x"}])
def test_a_malformed_api_ps_body_falls_back_to_the_ratio(ps: object) -> None:
    """A body that is not ``/api/ps``'s shape reads as no window, not a crash."""
    from openral_core.exceptions import ROSPlanningError
    from openral_reasoner.palette import ToolPalette

    replies = [_ollama_reply(4096, "fp_ollama")]
    with FakeLLMServer(replies, get_bodies={"/api/ps": ps}) as server:
        client = OpenAICompatibleToolUseClient(
            model_id="qwen3.5:4b", base_url=server.openai_base_url, tool_choice="auto"
        )
        with pytest.raises(ROSPlanningError, match="cut the prompt"):
            client.select_tool(context_text="x" * 31_000, palette=ToolPalette(skills=()))

"""Unit tests for the ``cosmos`` reasoner provider (NVIDIA Cosmos 3).

Covers the env-driven factory branch, the managed-server lifecycle decisions
(autostart on/off, loopback-only, endpoint-up short-circuit), sidecar script
resolution, and the tool-call wire path through
:class:`~openral_reasoner.cosmos3.Cosmos3ToolUseClient`.

§1.11 rule: the only doubles are the ``openai`` SDK object at the network
boundary (mirroring ``test_reasoner_invalid_plan_feedback.py``) and a
monkeypatched HTTP endpoint probe — both process/network boundaries. No
openral types are mocked; palettes are built with the real
:func:`build_tool_palette` against a real :class:`RobotCapabilities`.

Run with:
    uv run pytest tests/unit/test_reasoner_cosmos3.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from openral_core import EmitPromptTool, RobotCapabilities
from openral_core.exceptions import ROSConfigError
from openral_reasoner import cosmos3
from openral_reasoner.cosmos3 import (
    COSMOS3_BASE_URL,
    DEFAULT_COSMOS3_MODEL,
    Cosmos3ToolUseClient,
    find_cosmos3_sidecar_script,
)
from openral_reasoner.palette import ToolPalette, build_tool_palette
from openral_reasoner.tool_use import build_tool_use_client_from_env

_ENV_VARS = (
    "OPENRAL_REASONER_MODEL",
    "OPENRAL_REASONER_ENDPOINT",
    "OPENRAL_REASONER_API_KEY",
    "OPENRAL_REASONER_DIALECT",
    "OPENRAL_REASONER_MAX_TOKENS",
    "OPENRAL_REASONER_TIMEOUT_S",
    # Retired provider-first spelling (removed in 0.3.0); still cleared so a
    # stale developer env cannot influence a test.
    "OPENRAL_REASONER_LLM_PROVIDER",
    "OPENRAL_REASONER_LLM_MODEL",
    "OPENRAL_REASONER_LLM_API_KEY",
    "OPENRAL_REASONER_LLM_BASE_URL",
    "OPENRAL_REASONER_LLM_MAX_TOKENS",
    "OPENRAL_REASONER_LLM_TIMEOUT_S",
    "OPENRAL_COSMOS3_AUTOSTART",
    "OPENRAL_COSMOS3_BOOT_TIMEOUT_S",
    "OPENRAL_COSMOS3_SIDECAR",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip reasoner/cosmos env before every test so a stray developer
    env doesn't shadow the case under test."""
    for key in _ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def _empty_palette() -> ToolPalette:
    return build_tool_palette(
        installed_skills=[],
        robot_capabilities=RobotCapabilities(embodiment_tags=["franka_panda"]),
    )


# ── Factory branch ─────────────────────────────────────────────────────────────


def test_cosmos_builds_client_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """`cosmos3-edge` is a curated key: the registry supplies the served id."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client.model_id == DEFAULT_COSMOS3_MODEL == "nvidia/Cosmos3-Edge"
    assert client._base_url == COSMOS3_BASE_URL == "http://127.0.0.1:8901/v1"
    assert client._api_key is None
    # On-device VLM first-call headroom (kernel compilation after boot).
    assert client._timeout_s == 120.0
    assert client._auto_start is True
    assert client._tool_choice == "required"


def test_cosmos_explicit_endpoint_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote endpoint is selectable, and disengages the managed autostart.

    The served model id comes from the registry entry — pointing the managed
    client at a different Cosmos tier is a registry change (a new curated
    model), not an env one.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "http://thor.lan:8000/v1")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client.model_id == DEFAULT_COSMOS3_MODEL
    assert client._base_url == "http://thor.lan:8000/v1"
    # Non-loopback endpoint → the managed autostart must disengage.
    assert client._auto_start is False


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_cosmos_autostart_env_disables_spawn(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Every common falsy spelling disables the managed spawn — an operator
    who writes AUTOSTART=false must not get a surprise multi-GB server."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    monkeypatch.setenv("OPENRAL_COSMOS3_AUTOSTART", value)
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client._auto_start is False


def test_cosmos_portless_loopback_disables_autostart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loopback URL without an explicit port can't be managed: the spawned
    server and the readiness probe would target different ports (reverse-proxy
    on :80 case) — treat it as self-managed instead."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "http://127.0.0.1/v1")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client._auto_start is False


def test_cosmos_boot_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    monkeypatch.setenv("OPENRAL_COSMOS3_BOOT_TIMEOUT_S", "45")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client._boot_timeout_s == 45.0


def test_cosmos_timeout_env_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    monkeypatch.setenv("OPENRAL_REASONER_TIMEOUT_S", "15")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)
    assert client._timeout_s == 15.0


def test_curated_menu_lists_cosmos() -> None:
    """An unconfigured reasoner names cosmos3-edge among the curated models."""
    with pytest.raises(ROSConfigError) as excinfo:
        build_tool_use_client_from_env()
    assert "cosmos3-edge" in str(excinfo.value)


# ── Sidecar script resolution ──────────────────────────────────────────────────


def test_find_sidecar_script_resolves_in_tree() -> None:
    script = find_cosmos3_sidecar_script()
    assert script.name == "cosmos3_reasoner_sidecar.py"
    assert script.exists()


def test_find_sidecar_script_env_override_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "my_sidecar.py"
    override.write_text("# placeholder\n")
    monkeypatch.setenv("OPENRAL_COSMOS3_SIDECAR", str(override))
    assert find_cosmos3_sidecar_script() == override


def test_find_sidecar_script_missing_override_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale/typo'd override must fail with its real cause, not surface
    later as a misleading 'sidecar exited early' GPU/driver diagnosis."""
    monkeypatch.setenv("OPENRAL_COSMOS3_SIDECAR", str(tmp_path / "nope.py"))
    with pytest.raises(ROSConfigError) as excinfo:
        find_cosmos3_sidecar_script()
    assert "OPENRAL_COSMOS3_SIDECAR" in str(excinfo.value)
    assert "does not exist" in str(excinfo.value)


def test_spawn_argv_shape() -> None:
    """The spawn argv forwards model/host/port parsed from the base URL."""
    client = Cosmos3ToolUseClient(base_url="http://127.0.0.1:9333/v1", auto_start=True)
    argv = client._spawn_argv()
    assert argv[0] == sys.executable
    assert argv[1].endswith("cosmos3_reasoner_sidecar.py")
    assert argv[argv.index("--model") + 1] == DEFAULT_COSMOS3_MODEL
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "9333"


# ── Managed-server lifecycle decisions ─────────────────────────────────────────


def test_autostart_off_and_endpoint_down_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint down + autostart disabled → a config error naming the fix."""
    monkeypatch.setattr(cosmos3, "_endpoint_is_up", lambda *_a, **_k: False)
    client = Cosmos3ToolUseClient(auto_start=False)
    with pytest.raises(ROSConfigError) as excinfo:
        client.select_tool(context_text="go", palette=_empty_palette())
    msg = str(excinfo.value)
    assert "cosmos3_reasoner_sidecar" in msg
    assert "OPENRAL_REASONER_ENDPOINT" in msg


def test_endpoint_up_short_circuits_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live endpoint is used as-is — no sidecar spawn, straight to the SDK."""
    monkeypatch.setattr(cosmos3, "_endpoint_is_up", lambda *_a, **_k: True)

    def _boom(*_a: object, **_k: object) -> None:  # pragma: no cover — must not run
        raise AssertionError("spawn must not happen when the endpoint is up")

    monkeypatch.setattr(cosmos3.subprocess, "Popen", _boom)
    _install_fake_openai(
        monkeypatch,
        arguments=(
            '{"target_topic": "/openral/prompt", "text": "hello", "rationale": "next step"}'
        ),
    )
    client = Cosmos3ToolUseClient(auto_start=True)
    call = client.select_tool(context_text="go", palette=_empty_palette())
    assert isinstance(call, EmitPromptTool)
    assert call.text == "hello"
    # The probe result is cached — the next call must not re-probe per tick.
    assert client._server_ready is True


def test_live_child_is_reused_not_respawned(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a boot timeout the still-provisioning child must be waited on,
    never duplicated — a second Popen would orphan the first (leaked vLLM +
    port/VRAM fight)."""
    probes = {"n": 0}

    def _up(*_a: object, **_k: object) -> bool:
        probes["n"] += 1
        return probes["n"] >= 3  # down for the pre-check + first loop probe, then up

    monkeypatch.setattr(cosmos3, "_endpoint_is_up", _up)
    monkeypatch.setattr(cosmos3.time, "sleep", lambda _s: None)

    def _boom(*_a: object, **_k: object) -> None:  # pragma: no cover — must not run
        raise AssertionError("must not spawn a duplicate while the child is alive")

    monkeypatch.setattr(cosmos3.subprocess, "Popen", _boom)

    class _LiveChild:
        returncode = None

        def poll(self) -> None:
            return None

    client = Cosmos3ToolUseClient(auto_start=True, boot_timeout_s=30.0)
    client._child = _LiveChild()  # type: ignore[assignment]  # reason: process-boundary double per §1.11
    client._ensure_server()
    assert client._server_ready is True


def test_boot_timeout_keeps_child_for_next_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout must not discard the (possibly still-downloading) child —
    the next call waits on it instead of double-spawning."""
    monkeypatch.setattr(cosmos3, "_endpoint_is_up", lambda *_a, **_k: False)

    class _LiveChild:
        returncode = None

        def poll(self) -> None:
            return None

    child = _LiveChild()
    client = Cosmos3ToolUseClient(auto_start=True, boot_timeout_s=0.0)
    client._child = child  # type: ignore[assignment]  # reason: process-boundary double per §1.11
    with pytest.raises(ROSConfigError) as excinfo:
        client._ensure_server()
    assert "not ready within" in str(excinfo.value)
    assert client._child is child  # preserved, not replaced or dropped


def test_early_exit_child_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sidecar that dies during boot surfaces its exit code, not a hang."""
    monkeypatch.setattr(cosmos3, "_endpoint_is_up", lambda *_a, **_k: False)

    class _DeadChild:
        returncode = 3

        def poll(self) -> int:
            return 3

    monkeypatch.setattr(cosmos3.subprocess, "Popen", lambda *_a, **_k: _DeadChild())
    client = Cosmos3ToolUseClient(auto_start=True, boot_timeout_s=5.0)
    with pytest.raises(ROSConfigError) as excinfo:
        client.select_tool(context_text="go", palette=_empty_palette())
    assert "exited early" in str(excinfo.value)
    assert "code 3" in str(excinfo.value)


# ── Wire path (openai SDK network-boundary double, §1.11) ─────────────────────


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, *, arguments: str) -> None:
    """Patch ``openai.OpenAI`` so ``create()`` returns one ``emit_prompt`` call.

    Mirrors the real SDK response graph
    (``response.choices[0].message.tool_calls[0].function.{name,arguments}``).
    """

    def _create(**_kwargs: object) -> SimpleNamespace:
        function = SimpleNamespace(name="emit_prompt", arguments=arguments)
        tool_call = SimpleNamespace(function=function)
        message = SimpleNamespace(tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    import openai  # reason: network-boundary double per §1.11

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)


# ── Sidecar pre-warm ─────────────────────────────────────────────────────────


def test_warm_starts_the_sidecar_without_a_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """``warm()`` reaches ``_ensure_server`` — the whole point of the seam.

    Without it the sidecar boots lazily from ``select_tool``, i.e. on the
    reasoner's first tick, after the graph is already up and an operator is
    waiting. On a cold host that is a venv provision plus a ~9 GB download;
    bringup has minutes of unrelated work to overlap it with.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    client = build_tool_use_client_from_env()

    calls: list[int] = []
    monkeypatch.setattr(type(client), "_ensure_server", lambda _self: calls.append(1))

    client.warm()  # type: ignore[attr-defined]

    assert calls == [1]


def test_warm_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-warm plus the lazy path in select_tool must not double-spawn.

    ``_ensure_server`` short-circuits on ``_server_ready`` and reuses a
    still-booting child, so calling ``warm()`` more than once is safe.
    """
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    client = build_tool_use_client_from_env()

    calls: list[int] = []
    monkeypatch.setattr(type(client), "_ensure_server", lambda _self: calls.append(1))

    client.warm()  # type: ignore[attr-defined]
    client.warm()  # type: ignore[attr-defined]

    assert calls == [1, 1], "warm() delegates every time; _ensure_server owns the de-dup"


def test_concurrent_ensure_server_spawns_exactly_one_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warm() (llm pool) and describe_image (vlm pool) racing on a cold host
    must not double-spawn `vllm serve`.

    Regression: `_ensure_server`'s check-then-act on `_child`/`_server_ready`
    was unlocked, so two threads could both see `_child is None` and both
    Popen — two 4B servers fighting for the port and doubling VRAM, with the
    first handle orphaned. Doubles here are Popen + the HTTP probe, both
    process/network boundaries (§1.11).
    """
    import threading

    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")
    client = build_tool_use_client_from_env()
    assert isinstance(client, Cosmos3ToolUseClient)

    spawns: list[list[str]] = []
    barrier = threading.Barrier(2, timeout=10.0)

    class _FakeChild:
        def poll(self) -> None:
            return None  # still running

    def _fake_popen(argv: list[str]) -> _FakeChild:
        spawns.append(argv)
        return _FakeChild()

    # Endpoint is down until a spawn happened; up right after (instant boot).
    monkeypatch.setattr(cosmos3, "_endpoint_is_up", lambda _url: bool(spawns))
    monkeypatch.setattr(cosmos3.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(cosmos3.atexit, "register", lambda _fn: None)

    errors: list[BaseException] = []

    def _race() -> None:
        barrier.wait()  # maximise overlap of the two callers
        try:
            client._ensure_server()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_race) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors, f"_ensure_server raised under contention: {errors!r}"
    assert len(spawns) == 1, f"expected exactly one sidecar spawn, got {len(spawns)}"
    assert client._server_ready is True

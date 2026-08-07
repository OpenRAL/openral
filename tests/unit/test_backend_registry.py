"""Unit tests for the runtime-backend / policy-attach-hook registry.

Coverage
--------
- ``resolve_runtime_backend``: built-in names (``pytorch``, ``onnx``, ``null``)
  resolve to the real class; an unregistered name raises ``ROSConfigError``
  naming ``openral-pro-trt``; a name registered under the
  ``openral.runtime_backends`` entry-point group resolves to whatever real
  class the entry point targets (proves the discovery path, not just the
  built-in dict).
- ``maybe_attach_pro_hooks``: no hook installed → ``False`` (debug log, no
  raise); a registered ``openral.policy_attach_hooks`` entry point is invoked
  and its truthy/falsy return propagates.

Both entry-point tests monkeypatch ``entry_points`` with real
``importlib.metadata.EntryPoint`` objects pointing at real, importable
targets (an existing ``Runtime`` class; a module-level function in this
file) — the discovery machinery is exercised for real, only the "what's
installed" answer is controlled (CLAUDE.md §1.11: no mocks/stubs).
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

import pytest
from openral_core.exceptions import ROSConfigError
from openral_rskill.backend_registry import maybe_attach_pro_hooks, resolve_runtime_backend
from openral_rskill.runtime import NullRuntime
from openral_rskill.runtime_onnx import ONNXRuntime
from openral_rskill.runtime_pytorch import PyTorchRuntime


def _installed(group: str, name: str) -> bool:
    """True when the REAL (unpatched) entry-point group ships ``name``."""
    return any(ep.name == name for ep in entry_points(group=group))


# The two unpatched tests below assert the open-source-only shape of the
# environment. That holds in CI and on a plain checkout, but `openral-pro-trt`
# is installed on some dev boxes — there the assertions are not wrong, they are
# unobservable, so skip rather than fail (CLAUDE.md §1.11: an unavailable — or
# here, an unavoidably-present — dependency is a skip with a reason, never a
# faked result).
_PRO_TRT_BACKEND = pytest.mark.skipif(
    _installed("openral.runtime_backends", "tensorrt"),
    reason="openral-pro-trt is installed here; this test asserts the absent-backend branch",
)
_PRO_SMOLVLA_HOOK = pytest.mark.skipif(
    _installed("openral.policy_attach_hooks", "smolvla"),
    reason="openral-pro-trt is installed here; this test asserts the no-hook branch",
)


class TestResolveRuntimeBackendBuiltins:
    def test_pytorch(self) -> None:
        assert resolve_runtime_backend("pytorch") is PyTorchRuntime

    def test_onnx(self) -> None:
        assert resolve_runtime_backend("onnx") is ONNXRuntime

    def test_null(self) -> None:
        assert resolve_runtime_backend("null") is NullRuntime


class TestResolveRuntimeBackendMiss:
    def test_unregistered_name_raises_named_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No openral-pro-trt installed -> the miss path names it (CLAUDE.md §1.4)."""
        monkeypatch.setattr("openral_rskill.backend_registry.entry_points", lambda group: ())
        with pytest.raises(ROSConfigError, match="openral-pro-trt"):
            resolve_runtime_backend("tensorrt")

    @_PRO_TRT_BACKEND
    def test_real_environment_has_no_tensorrt_backend(self) -> None:
        """Without monkeypatching: a plain repo checkout has no openral-pro-trt
        installed, so the real, unpatched entry-point group is genuinely empty."""
        with pytest.raises(ROSConfigError, match="tensorrt"):
            resolve_runtime_backend("tensorrt")


class TestResolveRuntimeBackendEntryPoint:
    def test_registered_entry_point_resolves_real_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ep = EntryPoint(
            name="tensorrt",
            value="openral_rskill.runtime_pytorch:PyTorchRuntime",
            group="openral.runtime_backends",
        )
        monkeypatch.setattr(
            "openral_rskill.backend_registry.entry_points",
            lambda group: (ep,) if group == "openral.runtime_backends" else (),
        )
        assert resolve_runtime_backend("tensorrt") is PyTorchRuntime

    def test_builtin_wins_over_entry_point_of_same_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malicious/misconfigured entry point can't shadow a built-in name."""
        ep = EntryPoint(
            name="pytorch",
            value="openral_rskill.runtime_onnx:ONNXRuntime",
            group="openral.runtime_backends",
        )
        monkeypatch.setattr(
            "openral_rskill.backend_registry.entry_points",
            lambda group: (ep,) if group == "openral.runtime_backends" else (),
        )
        assert resolve_runtime_backend("pytorch") is PyTorchRuntime


def _fake_attach_hook(skill: object, **kwargs: object) -> bool:
    """Stand-in policy-attach hook — exercises the real entry-point load path."""
    return kwargs.get("device") == "cuda:0"


class TestMaybeAttachProHooks:
    @_PRO_SMOLVLA_HOOK
    def test_no_hook_installed_returns_false(self) -> None:
        assert maybe_attach_pro_hooks("smolvla", NullRuntime()) is False

    def test_registered_hook_declines_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = EntryPoint(
            name="smolvla",
            value=f"{__name__}:_fake_attach_hook",
            group="openral.policy_attach_hooks",
        )
        monkeypatch.setattr(
            "openral_rskill.backend_registry.entry_points",
            lambda group: (ep,) if group == "openral.policy_attach_hooks" else (),
        )
        assert maybe_attach_pro_hooks("smolvla", NullRuntime(), device="cpu") is False

    def test_registered_hook_attaches_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = EntryPoint(
            name="smolvla",
            value=f"{__name__}:_fake_attach_hook",
            group="openral.policy_attach_hooks",
        )
        monkeypatch.setattr(
            "openral_rskill.backend_registry.entry_points",
            lambda group: (ep,) if group == "openral.policy_attach_hooks" else (),
        )
        assert maybe_attach_pro_hooks("smolvla", NullRuntime(), device="cuda:0") is True

    def test_hook_name_mismatch_falls_through_to_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hook registered for 'act' does not answer a lookup for 'smolvla'."""
        ep = EntryPoint(
            name="act",
            value=f"{__name__}:_fake_attach_hook",
            group="openral.policy_attach_hooks",
        )
        monkeypatch.setattr(
            "openral_rskill.backend_registry.entry_points",
            lambda group: (ep,) if group == "openral.policy_attach_hooks" else (),
        )
        assert maybe_attach_pro_hooks("smolvla", NullRuntime(), device="cuda:0") is False

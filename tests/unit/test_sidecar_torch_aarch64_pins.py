"""Unit tests: no sidecar may pin a torch/triton version without aarch64 wheels.

Regression coverage for issue #88. PyTorch's ``cu128`` index publishes a
``manylinux_2_28_aarch64`` wheel for every recent release **except 2.8.0**, and
the ``triton`` that torch 2.8.0 requires (3.4.0) has no aarch64 wheel on any
index either. Several upstream VLA projects pin exactly that pair, so the
``xr1`` / ``lingbot_vla2`` / ``qwen_vlm`` / ``locateanything`` sidecars could
not provision at all on an aarch64 CUDA host (GB10 / DGX Spark, Jetson Thor):
``uv`` failed with "no wheels with a matching platform tag" minutes into the
install. Every pin set this repo owns now targets torch 2.9.x.

The forbidden versions are enumerated rather than range-checked because this is
a *wheel-availability* fact, not a compatibility floor: 2.7.x is older than
2.8.0 and fine on the index, 2.9.x is newer and fine, only 2.8.x is missing.
See ``docs/reference/aarch64-support.md`` for the full matrix.

Offline by construction: the pins are read from the real sidecar modules and
the real requirement/lock files, not resolved against the network (CLAUDE.md
§1.11 — real components, and a unit test may not depend on PyPI being up).

Run with:
    uv run pytest tests/unit/test_sidecar_torch_aarch64_pins.py -v
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "tools"
_REQS = _TOOLS / "sidecar_requirements"

#: Versions with no ``linux_aarch64`` wheel, as ``(package, version)``. Verified
#: against download.pytorch.org/whl/cu128 and PyPI on 2026-08-09.
_NO_AARCH64_WHEEL = {
    ("torch", "2.8.0"),
    ("torch", "2.8.1"),
    ("triton", "3.3.0"),
    ("triton", "3.3.1"),
    ("triton", "3.4.0"),
}

#: Sidecars whose torch pins this repo owns and which install from a CUDA index.
#: ``lingbot_vla2 --variant v1`` is deliberately absent: ``lerobot==0.4.2``
#: caps ``torch<2.8.0``, which rules out every version that has an aarch64
#: wheel, so V1 is x86_64-only until it moves off that lerobot.
_PIN_REQUIREMENT_FILES = ("qwen_vlm.in", "locateanything.in")
_PIN_LOCK_FILES = ("qwen_vlm.lock", "locateanything.lock")


def _load(name: str):  # type: ignore[no-untyped-def] # reason: importlib returns ModuleType via a dynamic spec, same idiom as test_cosmos3_sidecar.py
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pins(requirements: object) -> set[tuple[str, str]]:
    """Parse ``name==version`` pins out of a requirement string or iterable."""
    text = requirements if isinstance(requirements, str) else "\n".join(requirements)  # type: ignore[arg-type] # reason: callers pass str | Sequence[str]
    return {
        (m.group(1).lower(), m.group(2))
        for m in re.finditer(r"^([A-Za-z0-9_.-]+)==([0-9][^\s;\\+]*)", text, re.MULTILINE)
    }


def _assert_installable_on_aarch64(pins: set[tuple[str, str]], where: str) -> None:
    offenders = sorted(pins & _NO_AARCH64_WHEEL)
    assert not offenders, (
        f"{where} pins {offenders}, which publish no linux_aarch64 wheel — the "
        "sidecar cannot provision on a GB10 / Jetson Thor host (issue #88). "
        "See docs/reference/aarch64-support.md."
    )


def test_xr1_torch_pins_have_aarch64_wheels() -> None:
    xr1 = _load("xr1_sidecar")
    pins = _pins(xr1._XR1_TORCH_DEPS)
    _assert_installable_on_aarch64(pins, "tools/xr1_sidecar.py::_XR1_TORCH_DEPS")
    assert ("torch", "2.9.1") in pins, "the torch pin moved without updating this test"


def test_lingbot_v2_overrides_have_aarch64_wheels() -> None:
    lingbot = _load("lingbot_vla2_sidecar")
    pins = _pins(lingbot._V2_OVERRIDES)
    _assert_installable_on_aarch64(pins, "tools/lingbot_vla2_sidecar.py::_V2_OVERRIDES")
    assert ("torch", "2.9.1") in pins
    assert ("triton", "3.5.1") in pins


def test_lingbot_v2_overrides_cover_every_upstream_torch_pin() -> None:
    """The override set must name every torch-stack package upstream pins.

    ``uv pip install --overrides`` only replaces requirements it names. Upstream
    pins five coupled packages; missing one would let e.g. ``torchvision==0.23.0``
    (torch-2.8 ABI) back in next to torch 2.9.1 and break the extension load.
    """
    lingbot = _load("lingbot_vla2_sidecar")
    overridden = {name for name, _ in _pins(lingbot._V2_OVERRIDES)}
    assert overridden == {"torch", "torchvision", "torchaudio", "torchcodec", "triton"}


def test_internvla_n1_torch_pins_have_aarch64_wheels() -> None:
    """The InternVLA-N1 nav sidecar's torch pin must be a CUDA-capable one.

    Its failure mode was quieter than the others': ``torch==2.6.0`` with no
    ``--torch-backend`` resolved PyPI's aarch64 wheel, which is **CPU-only**, so
    the 8.3B dual-system model provisioned, booted and answered — on the CPU.
    The ``transformers==4.51.0`` bound the old pin comment cited is not real
    (4.51.0 declares ``torch>=2.0``, diffusers 0.32.2 ``torch>=1.4``).
    """
    ivn = _load("internvla_n1_sidecar")
    pins = _pins(ivn._PINNED_DEPS)
    _assert_installable_on_aarch64(pins, "tools/internvla_n1_sidecar.py::_PINNED_DEPS")
    assert ("torch", "2.9.1") in pins, "the torch pin moved without updating this test"
    assert ("torchvision", "0.24.1") in pins
    # The checkpoint-compatibility pin, not a torch one — see the file's comment.
    assert ("diffusers", "0.32.2") in pins, "diffusers 0.33 breaks the DualVLN NextDiT load"


@pytest.mark.parametrize("filename", _PIN_REQUIREMENT_FILES + _PIN_LOCK_FILES)
def test_sidecar_requirement_files_have_aarch64_wheels(filename: str) -> None:
    path = _REQS / filename
    _assert_installable_on_aarch64(_pins(path.read_text(encoding="utf-8")), str(path))


def test_nvrtc_override_covers_both_marker_branches() -> None:
    """The nvrtc override must pin every platform, not just aarch64.

    A uv override *replaces* every requirement for the package it names. An
    override listing only the ``== "aarch64"`` branch therefore leaves x86_64
    with no nvrtc requirement at all rather than falling back to torch's own
    pin — the first cut of this file did exactly that and silently dropped
    nvrtc from the x86_64 half of both locks.
    """
    pins = _pins((_REQS / "aarch64-nvrtc-override.txt").read_text(encoding="utf-8"))
    assert ("nvidia-cuda-nvrtc-cu12", "12.9.86") in pins, "aarch64 must get an sm_121-aware nvrtc"
    assert ("nvidia-cuda-nvrtc-cu12", "12.8.93") in pins, "non-aarch64 must keep torch's own pin"


@pytest.mark.parametrize("filename", _PIN_LOCK_FILES)
def test_locks_carry_both_nvrtc_branches(filename: str) -> None:
    """Both nvrtc branches must survive into the compiled lock.

    Regression guard for the same failure one level down: a lock recompiled
    without ``--overrides``, or with a one-sided override, installs the wrong
    nvrtc (or none) on one of the two platforms. The markers are what make a
    single universal lock serve both.
    """
    lock = (_REQS / filename).read_text(encoding="utf-8")
    for marker in ("platform_machine == 'aarch64'", "platform_machine != 'aarch64'"):
        assert re.search(rf"^nvidia-cuda-nvrtc-cu12==[\d.]+ ; {re.escape(marker)}", lock, re.M), (
            f"{filename} is missing the nvrtc branch for `{marker}`; recompile it with the "
            "`uv pip compile … --overrides` command in the .in file's header."
        )


@pytest.mark.skipif(shutil.which("uv") is None, reason="ensure_pip_venv provisions through uv")
def test_every_xr1_install_pass_carries_the_nvrtc_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-pass sidecar must pass ``--overrides`` on *every* uv pass.

    uv re-resolves the whole environment on each ``pip install``. A pass that
    omits the override file sees torch's exact ``nvidia-cuda-nvrtc-cu12==12.8.93``
    pin again and silently downgrades the shim an earlier pass installed, so the
    fix survives exactly one command and inference goes back to dying in nvrtc.
    That regression was observed live in the LingBot sidecar's extras pass; XR-1
    has the same three-pass shape.

    ``run_cmd`` is the module's subprocess boundary, which §1.11 allows doubling;
    the venv itself is really provisioned by the un-patched ``ensure_pip_venv``.
    """
    xr1 = _load("xr1_sidecar")
    calls: list[list[str]] = []
    monkeypatch.setattr(xr1, "run_cmd", lambda label, cmd, **kw: calls.append(list(cmd)))
    xr1._ensure_venv(tmp_path)

    installs = [c for c in calls if "install" in c]
    assert len(installs) == 3, f"expected torch/runtime/attn passes, got {len(installs)}"
    for cmd in installs:
        assert "--overrides" in cmd, f"uv pass without the nvrtc override: {' '.join(cmd)}"


@pytest.mark.skipif(shutil.which("uv") is None, reason="ensure_pip_venv provisions through uv")
def test_every_lingbot_v2_install_pass_carries_the_nvrtc_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that actually shipped: LingBot's extras pass dropped it."""
    lingbot = _load("lingbot_vla2_sidecar")
    source = tmp_path / "source"
    source.mkdir()
    (source / "requirements.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(lingbot, "run_cmd", lambda label, cmd, **kw: calls.append(list(cmd)))
    lingbot._ensure_venv(tmp_path / "home", source)

    installs = [c for c in calls if "install" in c]
    assert len(installs) == 2, f"expected the requirements + extras passes, got {len(installs)}"
    for cmd in installs:
        assert str(lingbot._NVRTC_OVERRIDE) in cmd, (
            f"uv pass without the nvrtc override: {' '.join(cmd)}"
        )


def test_every_internvla_n1_install_pass_carries_the_cuda_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four InternVLA-N1 passes need ``--torch-backend`` *and* the override.

    This sidecar installs in four passes (pins, ``diffusion_policy``, the
    editable ``internnav``, bitsandbytes). uv re-resolves on each one, so a pass
    missing ``--torch-backend=cu128`` can swap the CUDA torch back for PyPI's
    aarch64 CPU build, and a pass missing the override lets torch's exact
    ``nvidia-cuda-nvrtc-cu12==12.8.93`` pin downgrade the sm_121-aware shim.
    """
    ivn = _load("internvla_n1_sidecar")
    source = tmp_path / "checkout"
    source.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(ivn, "run_cmd", lambda label, cmd, **kw: calls.append(list(cmd)))
    ivn._install_deps(source=source, uv="uv", quantization="nf4")

    installs = [c for c in calls if "install" in c]
    assert len(installs) == 4, f"expected pins/diffusion_policy/editable/bnb, got {len(installs)}"
    for cmd in installs:
        assert "--torch-backend=cu128" in cmd, f"uv pass without the CUDA index: {' '.join(cmd)}"
        assert str(ivn._NVRTC_OVERRIDE) in cmd, (
            f"uv pass without the nvrtc override: {' '.join(cmd)}"
        )


@pytest.mark.parametrize("filename", _PIN_LOCK_FILES)
def test_locks_match_their_source_torch_pin(filename: str) -> None:
    """A regenerated lock must carry the ``.in`` file's torch version.

    Guards the failure mode where the ``.in`` is bumped but the hash-pinned lock
    — which is what actually gets installed — is left behind.
    """
    lock = (_REQS / filename).read_text(encoding="utf-8")
    source = (_REQS / filename.replace(".lock", ".in")).read_text(encoding="utf-8")
    (wanted,) = [v for name, v in _pins(source) if name == "torch"]
    assert re.search(rf"^torch=={re.escape(wanted)}\+cu\d+", lock, re.MULTILINE), (
        f"{filename} does not pin torch=={wanted}; regenerate it with the "
        "`uv pip compile` command in the .in file's header."
    )

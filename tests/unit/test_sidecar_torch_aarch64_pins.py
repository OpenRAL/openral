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


@pytest.mark.parametrize("filename", _PIN_REQUIREMENT_FILES + _PIN_LOCK_FILES)
def test_sidecar_requirement_files_have_aarch64_wheels(filename: str) -> None:
    path = _REQS / filename
    _assert_installable_on_aarch64(_pins(path.read_text(encoding="utf-8")), str(path))


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

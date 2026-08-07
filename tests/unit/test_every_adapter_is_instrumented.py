"""Canary: every VLA adapter reaches the `inference_span` seam.

`openral.inference.duration` is emitted *only* by `inference_span`.
`InferenceRunnerBase` used to record it too, from `Skill.step` wall-time, but
that measured the chunk *dispatch* cost rather than the inference — the two
disagreed on chunked adapters and doubled the sample count — so it was removed.

A single seam is only safe while it is universal: an adapter that never opens
the span now emits no inference latency at all, silently. Five sidecar adapters
(behavior_groot, internvla_n1, lingbot_va_a1, lingbot_vla2, rlbench_3dda) were
exactly that gap until 2026-08-04, and the runner's record had been masking it.
This canary is source-level so the next adapter cannot reintroduce it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_POLICIES = _REPO_ROOT / "python" / "sim" / "src" / "openral_sim" / "policies"

# Not policy adapters: package glue, shared helpers, and the deterministic
# test double (which must stay dependency-free).
_NOT_ADAPTERS = {
    "__init__.py",
    "robots.py",
    "mock.py",
    "_processors.py",
    "_policy_loading.py",
    "_video_capture.py",
}

# The three ways to reach the seam: open the span directly, go through
# `run_inference` (which opens it for you), or hand the chunk forward to
# `build_chunk_executor`, whose `ChunkedExecutor` calls `run_inference`
# internally (openral_rskill/executor.py:_forward). The executor route is why
# gr00t / molmoact2 / openvla / pi05 / smolvla / xvla name no span symbol of
# their own yet are fully instrumented — a two-name check reported the first
# three as gaps they are not.
_SEAM_NAMES = {"inference_span", "run_inference", "build_chunk_executor"}


def _adapter_modules() -> list[pathlib.Path]:
    return sorted(p for p in _POLICIES.glob("*.py") if p.name not in _NOT_ADAPTERS)


@pytest.mark.parametrize("path", _adapter_modules(), ids=lambda p: p.name)
def test_adapter_reaches_the_inference_seam(path: pathlib.Path) -> None:
    """A module defining `step()` must call `inference_span` or `run_inference`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "step" for node in ast.walk(tree)
    ):
        pytest.skip(f"{path.name} defines no step() — not an inference adapter")

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called & _SEAM_NAMES, (
        f"{path.name} defines step() but never calls {sorted(_SEAM_NAMES)}; its "
        "inference emits no openral.inference.duration and no "
        "rskill.chunk_inference span"
    )


def test_the_canary_covers_every_shipped_adapter() -> None:
    """Guard the guard: the skip path must not quietly swallow the whole suite."""
    instrumented = [
        p
        for p in _adapter_modules()
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "step"
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
        )
    ]
    assert len(instrumented) >= 14, (
        f"only {len(instrumented)} adapters have a step(); the discovery glob or "
        "the _NOT_ADAPTERS exclusion list has drifted"
    )


def test_every_adapter_releases_all_its_module_fields() -> None:
    """`close()` must drop every torch-module field, not just `_policy`.

    `release_torch_modules` exists because `empty_cache()` only returns
    *already-free* blocks — a field the adapter still references pins its VRAM.
    xVLA released only `_policy` while holding four on-device lerobot
    processors (`_env_pre`/`_policy_pre`/`_policy_post`/`_env_post`), so its
    normalizer buffers survived every skill swap.
    """
    import re

    # Dataclass fields that hold a torch module tree and therefore must appear
    # in the adapter's `release_torch_modules(...)` call.
    module_field = re.compile(r"^\s{4}(_(?:policy|model|env|processor)\w*):\s", re.MULTILINE)

    offenders: list[str] = []
    for path in _adapter_modules():
        src = path.read_text(encoding="utf-8")
        call = re.search(r"release_torch_modules\((.*?)\n\s*\)", src, re.DOTALL)
        if call is None:
            continue  # adapters with no torch modules to release (sidecars)
        released = set(re.findall(r'"(_\w+)"', call.group(1)))
        declared = {m.group(1) for m in module_field.finditer(src)}
        missing = declared - released
        if missing:
            offenders.append(f"{path.name}: declares {sorted(missing)} but never releases them")

    assert not offenders, "adapters pin VRAM across a skill swap:\n  " + "\n  ".join(offenders)

"""Canary: every VLA adapter reaches the `inference_span` seam.

`openral.inference.duration` comes only from `inference_span`.
`InferenceRunnerBase` used to also record it from `Skill.step` wall-time —
measuring dispatch not inference, doubling the sample count on chunked
adapters — so it was removed.

Five sidecar adapters (behavior_groot, internvla_n1, lingbot_va_a1,
lingbot_vla2, rlbench_3dda) silently missed the seam until 2026-08-04,
masked by the runner's record. Source-level canary so it can't regress.
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

# Three ways to reach the seam: open the span directly, call `run_inference`
# (opens it), or hand the chunk to `build_chunk_executor` (its
# `ChunkedExecutor` calls `run_inference` internally,
# openral_rskill/executor.py:_forward) — why gr00t/molmoact2/openvla/pi05/
# smolvla/xvla name no span symbol yet are fully instrumented; a two-name
# check flagged the first three as false gaps.
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

    `release_torch_modules` exists because `empty_cache()` only frees
    already-free blocks — a still-referenced field pins VRAM. xVLA released only
    `_policy` while holding four lerobot processors (`_env_pre`/`_policy_pre`/
    `_policy_post`/`_env_post`), so normalizer buffers survived every skill swap.
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

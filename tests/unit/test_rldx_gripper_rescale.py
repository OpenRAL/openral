"""Unit tests for the RLDX → LIBERO gripper-action rescaling (GH-133).

The RLDX policy emits ``action.gripper`` in the RLDS dataset convention
(``[0, 1]`` with ``0=close`` / ``1=open``) — its training data was
standardized into that range. LIBERO's robosuite-based OSC controller
consumes ``[-1, +1]`` with the **opposite** sign convention (``-1=open``
/ ``+1=close``). Without a rescale step, the openral adapter feeds raw
``~0`` values into ``LiberoEnv.step``, the gripper never actuates, and
pick-and-place tasks deterministically fail. Reproduced as GH-133.

These tests exercise the real adapter helper
(:func:`openral_sim.policies.rldx._rldx_gripper_to_libero`) and the
real :meth:`_RLDXSidecarAdapter._assemble_libero_chunk` path with the
exact wire shape the upstream RLDX server emits ``(1, T=16, 1)`` —
mirrors ``rldx/policy/rldx_policy.py`` LIBERO-flat output. No mocks,
no stubs (CLAUDE.md §1.11 / §5.4).
"""

from __future__ import annotations

import numpy as np
import pytest

# Importing the rldx module triggers the @POLICIES.register("rldx")
# side-effect; we want that for the chunk-assembly integration test.
# The module-level imports themselves do not require pyzmq/msgpack — only
# adapter __post_init__ does — so we can probe the pure-numpy helpers
# without the opt-in group.
from openral_sim.policies.rldx import (
    _RLDX_ACTION_AXES,
    _rldx_gripper_to_libero,
    _RLDXSidecarAdapter,
)

# ─── helper: pure-numpy gripper rescale ──────────────────────────────────


@pytest.mark.parametrize(
    ("rlds_value", "expected_libero"),
    [
        # RLDS 0 = close → LIBERO +1 = close
        (0.0, 1.0),
        # RLDS 1 = open → LIBERO -1 = open
        (1.0, -1.0),
        # Anything > 0.5 binarizes to "open" (-1)
        (0.7, -1.0),
        (0.51, -1.0),
        # Anything < 0.5 binarizes to "close" (+1)
        (0.3, 1.0),
        (0.49, 1.0),
        # Truly neutral input → zero (no command); preserves
        # `sign(0)==0` semantics of an indecisive policy.
        (0.5, 0.0),
    ],
)
def test_rldx_gripper_to_libero_endpoints(rlds_value: float, expected_libero: float) -> None:
    """Confirms the upstream two-step transform on scalar inputs.

    The reference path in
    ``rldx/eval/sim/LIBERO/libero_env.py`` does
    ``invert(normalize(g)) = -sign(2g - 1)``. The openral helper must
    produce identical outputs so the LIBERO finetune sees its training
    distribution at inference time.
    """
    out = _rldx_gripper_to_libero(np.asarray([rlds_value], dtype=np.float32))
    assert out.shape == (1,)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(expected_libero)


def test_rldx_gripper_to_libero_chunk_shape() -> None:
    """Vectorised over a 16-step chunk: same rescale, preserved shape."""
    # Mix of open / close commands across a chunk.
    rlds = np.array([0.0, 0.1, 0.4, 0.49, 0.5, 0.51, 0.6, 0.9, 1.0] * 2, dtype=np.float32)[:16]
    out = _rldx_gripper_to_libero(rlds)
    assert out.shape == (16,)
    # Every entry must be in {-1, 0, +1}.
    assert set(np.unique(out).tolist()).issubset({-1.0, 0.0, 1.0})
    # First entry (RLDS 0 = close) → LIBERO +1.
    assert out[0] == 1.0
    # Last entry (RLDS 1 = open) → LIBERO -1.
    assert out[-1] == -1.0


def test_rldx_gripper_to_libero_does_not_mutate_input() -> None:
    """The helper returns a fresh array — caller's input is not aliased."""
    rlds = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    original = rlds.copy()
    _rldx_gripper_to_libero(rlds)
    np.testing.assert_array_equal(rlds, original)


# ─── integration: full LIBERO chunk assembly with the real adapter ───────


def _wire_action_dict(
    *,
    chunk_len: int = 16,
    gripper_values: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the on-the-wire action dict the upstream server returns.

    Layout matches ``RLDXSimPolicyWrapper`` (LIBERO suite, the path our
    adapter drives via ``--use-sim-policy-wrapper``):

        action.x / y / z / roll / pitch / yaw / gripper : (1, T, 1) float32

    See the module-docstring of ``openral_sim.policies.rldx`` for the
    full wire contract. The motion axes get small deterministic values
    so we can also verify the assembler preserves them untouched.
    """
    if gripper_values is None:
        gripper_values = np.linspace(0.0, 1.0, chunk_len, dtype=np.float32)
    if gripper_values.shape != (chunk_len,):
        raise AssertionError(f"test bug: bad gripper shape {gripper_values.shape}")
    out: dict[str, np.ndarray] = {}
    for i, axis in enumerate(_RLDX_ACTION_AXES[:-1]):  # all but gripper
        # 0.01, 0.02, ... per axis to make any column-swap visible.
        out[f"action.{axis}"] = np.full(
            (1, chunk_len, 1), fill_value=0.01 * (i + 1), dtype=np.float32
        )
    out["action.gripper"] = gripper_values.reshape(1, chunk_len, 1).astype(np.float32)
    return out


def test_assemble_libero_chunk_rescales_gripper_column() -> None:
    """End-to-end: server-shape input → 7-D LIBERO chunk with rescaled gripper.

    Exercises ``_RLDXSidecarAdapter._assemble_libero_chunk`` on real
    server-shaped data. Asserts:

    * the chunk is ``(T, 7)`` float32,
    * non-gripper axes survive unchanged (so the rescale only touches
      the gripper column),
    * the gripper column is in ``{-1, 0, +1}`` and obeys the upstream
      ``-sign(2g - 1)`` rule.
    """
    # Construct an adapter shell without invoking ``__post_init__`` —
    # the chunk-assembly path is pure numpy and does not need a live
    # ZMQ socket or sidecar.
    adapter = _RLDXSidecarAdapter.__new__(_RLDXSidecarAdapter)

    # Hand-pick gripper values that span the binarization boundary.
    grippers = np.array([0.0, 0.1, 0.49, 0.5, 0.51, 0.9, 1.0] + [0.0] * 9, dtype=np.float32)
    wire = _wire_action_dict(chunk_len=16, gripper_values=grippers)
    chunk = adapter._assemble_libero_chunk(wire)

    assert chunk.shape == (16, 7)
    assert chunk.dtype == np.float32

    # Non-gripper axes (indices 0..5) must equal the per-axis constants
    # we filled in (0.01, 0.02, ..., 0.06) at every step.
    for i in range(6):
        expected_val = 0.01 * (i + 1)
        np.testing.assert_allclose(chunk[:, i], expected_val, rtol=1e-6)

    # Gripper column (index 6) must be the rescaled values.
    expected_gripper = _rldx_gripper_to_libero(grippers)
    np.testing.assert_array_equal(chunk[:, 6], expected_gripper)
    # Every value must land in {-1, 0, +1}.
    assert set(np.unique(chunk[:, 6]).tolist()).issubset({-1.0, 0.0, 1.0})

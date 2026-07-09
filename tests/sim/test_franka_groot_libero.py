"""Sim-tier test: in-process GR00T N1.7 (NF4) on a LIBERO franka observation.

Replaces the retired Python-3.10 ZMQ sidecar with the native lerobot
``GrootPolicy`` backend. This exercises the **real** path end to end
— real ``rskills/gr00t-n17-libero`` manifest, real ``@POLICIES.register("gr00t")``
factory, real NF4 quantized GR00T checkpoint, real lerobot GR00T pre/post
processors — and asserts the backend emits a valid 7-D LIBERO action on a real
LIBERO franka observation.

Per CLAUDE.md §1.11 there are no mocks: the model, weights, quantization, and
processors are all real. The only double is at the sensor boundary (the camera
frames are seeded arrays), which is the sanctioned recorded-sensor exception.

GPU: loads a ~3 B GR00T model NF4 on one 8 GB card. Run under the shared GPU
lock: ``flock /tmp/openral-gpu.lock -c './.venv/bin/pytest -q
tests/sim/test_franka_groot_libero.py'``. Skips only when torch/CUDA or the
GR00T weights are genuinely unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCENE = _REPO_ROOT / "scenes" / "sim" / "libero_spatial.yaml"
_RSKILL = "rskills/gr00t-n17-libero"

_GR00T_LIBERO_ACTION_DIM = 7
_LIBERO_STATE_DIM = 8


def _fixed_libero_obs() -> dict[str, object]:
    """A deterministic in-distribution LIBERO franka observation.

    Two 256×256 RGB frames (seeded) + the checkpoint's own ``libero_sim`` 8-D
    state means (eef_pos(3) + axisangle(3) + gripper_qpos(2)).
    """
    rng = np.random.default_rng(1234)
    agent = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    state = np.array(
        [-0.02446, 0.10653, 1.05805, 3.06285, -0.10464, 0.08307, 0.01995, -0.02016],
        dtype=np.float32,
    )
    return {
        "images": {"camera1": agent, "camera2": wrist},
        "state": state,
        "task": "pick up the black bowl and place it on the plate",
    }


@pytest.fixture(scope="module")
def groot_adapter():
    """Build the in-process GR00T NF4 adapter, or skip if deps/weights absent."""
    torch = pytest.importorskip("torch", reason="GR00T needs torch")
    pytest.importorskip("lerobot", reason="GR00T needs lerobot")
    pytest.importorskip("bitsandbytes", reason="in-process GR00T NF4 needs bitsandbytes")
    # lerobot[groot] runtime extras (action head + prepare_input); skip cleanly
    # when the optional group is not installed (CLAUDE.md §1.11).
    pytest.importorskip("diffusers", reason="GR00T action head needs lerobot[groot]'s diffusers")
    pytest.importorskip("tree", reason="GR00T prepare_input needs lerobot[groot]'s dm-tree")
    if not torch.cuda.is_available():
        pytest.skip("GR00T NF4 requires a CUDA GPU")

    from tests.sim.conftest import compose_sim_env

    env = compose_sim_env(_SCENE, _RSKILL, n_episodes=1, max_steps=1)
    from openral_sim.policies import gr00t as _g  # noqa: F401 — register side effect
    from openral_sim.registry import POLICIES

    try:
        adapter = POLICIES.get("gr00t")(env)
    except ImportError as exc:
        # A missing lerobot[groot] extra (diffusers / dm-tree / decord) is an
        # environment gap, not a bug (CLAUDE.md §1.11).
        pytest.skip(f"lerobot[groot] extra unavailable: {exc}")
    except Exception as exc:
        # A gated/un-fetchable backbone (Cosmos-Reason2) is a legitimate skip,
        # not a failure — everything else is a real bug and must surface.
        msg = str(exc).lower()
        if any(tok in msg for tok in ("gated", "401", "403", "not found", "access")):
            pytest.skip(f"GR00T weights/backbone not fetchable here: {exc}")
        raise
    yield adapter
    adapter.close()


def test_groot_libero_emits_valid_action_chunk(groot_adapter) -> None:
    """The NF4 in-process backend returns a finite 7-D LIBERO action per step.

    Steps the adapter across a full execution horizon so the internal action
    queue both fills (first call → chunk inference) and drains (subsequent calls
    → popleft), asserting every emitted action is a valid 7-D vector with a
    gripper column inside the LIBERO ``[-1, 1]`` convention.
    """
    groot_adapter.reset()
    obs = _fixed_libero_obs()

    actions = [groot_adapter.step(obs, str(obs["task"])) for _ in range(8)]

    for i, action in enumerate(actions):
        assert action.shape == (_GR00T_LIBERO_ACTION_DIM,), (
            f"step {i}: expected {_GR00T_LIBERO_ACTION_DIM}-D LIBERO action, got {action.shape}"
        )
        assert np.all(np.isfinite(action)), f"step {i}: non-finite action {action}"
        # Gripper (LIBERO decode transform) must land in the robosuite [-1, 1] band.
        assert -1.0001 <= float(action[6]) <= 1.0001, f"step {i}: gripper {action[6]} out of range"

    # The action queue actually advanced (not the identical cached vector every
    # step) once it drained past the first inference.
    assert not np.allclose(actions[0], actions[-1]) or len(actions) == 1

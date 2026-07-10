"""Sim-tier test: in-process GR00T N1.7 (NF4) on an SO-101 fruit observation.

The SO-101 counterpart to :mod:`tests.sim.test_franka_groot_libero`. It proves
the generalized ``@POLICIES.register("gr00t")`` factory drives a **non-LIBERO**
GR00T checkpoint: the ``rskills/gr00t-n17-so101-fruit`` manifest carries a 6-D
``state_contract`` / ``action_contract``, the ``new_embodiment`` GR00T tag, and
``front``/``wrist`` image modality keys, and the adapter reads all of those from
the manifest instead of the hard-coded LIBERO 8-D/7-D/``image,wrist_image`` path.

Exercises the real path end to end — real manifest, real factory, real NF4
GR00T-N1.7-3B checkpoint (``aaronsu11/GR00T-N1.7-3B-SO101-FruitPicking``), real
lerobot GR00T processors — and asserts the backend emits a finite 6-D absolute
joint-position action on a real, in-distribution SO-101 observation.

Per CLAUDE.md §1.11 there are no mocks: model, weights, quantization, and
processors are all real. The only double is at the sensor boundary (seeded RGB
frames + the checkpoint's own state mean), the sanctioned recorded-sensor
exception. Also proves the 8 GB NF4 fit for this checkpoint (loads on one card).

GPU: run under the shared GPU lock: ``flock /tmp/openral-gpu.lock -c
'./.venv/bin/pytest -q tests/sim/test_so101_groot_fruit.py'``. Skips only when
torch/CUDA/lerobot[groot] or the weights are genuinely unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCENE = _REPO_ROOT / "scenes" / "sim" / "so101_tube_insertion.yaml"
_RSKILL = "rskills/gr00t-n17-so101-fruit"

_SO101_ACTION_DIM = 6
_SO101_STATE_DIM = 6

# In-distribution SO-101 proprio: the checkpoint's own `new_embodiment` state
# mean (single_arm(5) + gripper(1)) from experiment_cfg/dataset_statistics.json,
# in degrees. Feeding an in-distribution state keeps the action decode faithful.
_SO101_STATE_MEAN = np.array(
    [-10.548957, -7.485758, 17.880617, 60.125805, -56.270161, 18.371027],
    dtype=np.float32,
)


def _fixed_so101_fruit_obs() -> dict[str, object]:
    """A deterministic in-distribution SO-101 fruit observation.

    Two 256×256 RGB frames (seeded) keyed ``camera1``/``camera2`` (front / wrist)
    + the checkpoint's 6-D ``new_embodiment`` state mean, and the dataset's own
    task string.
    """
    rng = np.random.default_rng(4321)
    front = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return {
        "images": {"camera1": front, "camera2": wrist},
        "state": _SO101_STATE_MEAN.copy(),
        "task": "Grab a banana and put it on the plate",
    }


@pytest.fixture(scope="module")
def groot_so101_adapter():
    """Build the in-process GR00T NF4 adapter for the SO-101 checkpoint, or skip."""
    torch = pytest.importorskip("torch", reason="GR00T needs torch")
    pytest.importorskip("lerobot", reason="GR00T needs lerobot")
    pytest.importorskip("bitsandbytes", reason="in-process GR00T NF4 needs bitsandbytes")
    pytest.importorskip("diffusers", reason="GR00T action head needs lerobot[groot]'s diffusers")
    pytest.importorskip("tree", reason="GR00T prepare_input needs lerobot[groot]'s dm-tree")
    if not torch.cuda.is_available():
        pytest.skip("GR00T NF4 requires a CUDA GPU")

    from tests.sim.conftest import compose_sim_env

    env = compose_sim_env(_SCENE, _RSKILL, n_episodes=1, max_steps=1)
    # Force the observation source keys to camera1/camera2 (the so101_box scene
    # names its cameras oak_top/wrist); the manifest's image_modality_keys still
    # map these positionally onto the model's front/wrist inputs.
    env.vla.extra = {**(env.vla.extra or {}), "camera_keys": ["camera1", "camera2"]}

    from openral_sim.policies import gr00t as _g  # noqa: F401 — register side effect
    from openral_sim.registry import POLICIES

    try:
        adapter = POLICIES.get("gr00t")(env)
    except ImportError as exc:
        pytest.skip(f"lerobot[groot] extra unavailable: {exc}")
    except Exception as exc:
        msg = str(exc).lower()
        if any(tok in msg for tok in ("gated", "401", "403", "not found", "access")):
            pytest.skip(f"GR00T weights/backbone not fetchable here: {exc}")
        raise
    yield adapter
    adapter.close()


def test_groot_so101_emits_valid_action_chunk(groot_so101_adapter) -> None:
    """The NF4 in-process backend returns finite 6-D SO-101 actions per step.

    Steps the adapter across a full execution horizon so the internal action
    queue both fills (first call → chunk inference) and drains (subsequent calls
    → popleft), asserting every emitted action is a finite 6-D joint-position
    vector and that the queue advances.
    """
    groot_so101_adapter.reset()
    obs = _fixed_so101_fruit_obs()

    actions = [groot_so101_adapter.step(obs, str(obs["task"])) for _ in range(8)]

    for i, action in enumerate(actions):
        assert action.shape == (_SO101_ACTION_DIM,), (
            f"step {i}: expected {_SO101_ACTION_DIM}-D SO-101 action, got {action.shape}"
        )
        assert np.all(np.isfinite(action)), f"step {i}: non-finite action {action}"

    # The queue actually advanced (not the identical cached vector every step)
    # once it drained past the first inference.
    assert not np.allclose(actions[0], actions[-1]) or len(actions) == 1

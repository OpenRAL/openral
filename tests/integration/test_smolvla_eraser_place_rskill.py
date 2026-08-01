"""Real-checkpoint acceptance test for ``rskill-smolvla-so101-eraser_place-bf16``.

The manifest makes three claims that no schema check can verify — that the
declared camera aliasing routes the right view into the right checkpoint input,
that ``action_contract.joint_units`` is really ``degrees``, and that the chunk
shape / contract dims match the checkpoint. This test replays a REAL frame from
the checkpoint's own training dataset through the manifest-declared wiring and
compares the emitted chunk against the recorded teleop actions.

No mocks (CLAUDE.md §1.11): real manifest, real weights, real LeRobot dataset.
Skipped when ``lerobot`` / its ``dataset`` extra is absent or the Hub is
unreachable — never faked.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pytest
from openral_core.schemas import RSkillManifest

_Chunk: TypeAlias = "np.ndarray[Any, np.dtype[np.float32]]"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "rskills" / "rskill-smolvla-so101-eraser_place-bf16" / "rskill.yaml"

# The frame the assertions below are calibrated on: mid-episode, gripper closing
# on the eraser, so the ground-truth chunk carries real motion in every joint.
_FRAME = 100

# Per-joint MAE ceiling over the full 50-step chunk, in lerobot/degree units.
# Measured 0.15-1.39 with the manifest's wiring; the same frame with the two
# camera aliases SWAPPED measures 1.04-44.27, so this threshold is what
# separates "correctly wired" from "plausible but wrong".
_MAE_CEILING = 5.0


@pytest.fixture(scope="module")
def manifest() -> RSkillManifest:
    return RSkillManifest.from_yaml(str(_MANIFEST_PATH))


@pytest.fixture(scope="module")
def rollout(manifest: RSkillManifest) -> tuple[_Chunk, _Chunk]:
    """Return ``(predicted_chunk, ground_truth_chunk)`` for one real frame."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("lerobot", reason="lerobot not installed")
    pytest.importorskip("datasets", reason="lerobot[dataset] extra not installed")
    pytest.importorskip("transformers", reason="lerobot[smolvla] extra not installed")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    assert manifest.weights_uri is not None
    assert manifest.dataset_uri is not None
    assert manifest.image_preprocessing is not None

    # Honour a busy GPU: this checkpoint is 450 M params and the rest of the
    # suite (or another deploy) may already own the card. CPU is slow but
    # correct, and the assertions are device-independent.
    device = os.environ.get("OPENRAL_TEST_DEVICE") or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    repo_id = manifest.weights_uri.removeprefix("hf://")
    try:
        dataset = LeRobotDataset(manifest.dataset_uri.removeprefix("hf://"))
        config = PreTrainedConfig.from_pretrained(repo_id)
        config.device = device
        policy = SmolVLAPolicy.from_pretrained(repo_id, config=config).to(device).eval()
    except OSError as exc:  # network down / repo gated — skip, never fake
        pytest.skip(f"cannot reach the Hub for {repo_id!r}: {exc}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=repo_id,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    sample = dataset[_FRAME]
    # Replay the deploy wiring: the runner re-keys the robot's sensors onto the
    # camera1/camera2 slots via robots/so101_follower's `vla_feature_key`
    # (top -> camera1, wrist -> camera2), then the manifest's aliases rename
    # those slots to the checkpoint's own input-feature keys.
    aliases = manifest.image_preprocessing.aliases
    batch = {
        "observation.state": sample["observation.state"][None].to(device),
        f"observation.images.{aliases['camera1']}": (
            sample["observation.images.front"][None].to(device)
        ),
        f"observation.images.{aliases['camera2']}": (
            sample["observation.images.wrist"][None].to(device)
        ),
        "task": sample["task"],
    }
    assert not set(policy.config.image_features) - set(batch), (
        "the manifest's aliases do not cover every checkpoint image feature"
    )

    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(preprocessor(batch)))

    predicted: _Chunk = chunk[0].float().cpu().numpy()
    truth: _Chunk = np.stack(
        [dataset[_FRAME + i]["action"].numpy() for i in range(manifest.chunk_size)]
    )
    return predicted, truth


def test_manifest_matches_checkpoint_task(manifest: RSkillManifest) -> None:
    """The single training task string must survive verbatim into the description."""
    assert "place the erase on the blue square" in manifest.description


def test_chunk_shape_matches_contract(
    manifest: RSkillManifest, rollout: tuple[_Chunk, _Chunk]
) -> None:
    assert manifest.action_contract is not None
    predicted, _ = rollout
    assert predicted.shape == (manifest.chunk_size, manifest.action_contract.dim)


def test_actions_are_on_the_declared_degrees_scale(
    manifest: RSkillManifest, rollout: tuple[_Chunk, _Chunk]
) -> None:
    """Verify the declared joint units against the checkpoint's real output.

    A radians checkpoint could never exceed ~pi; this one must, or the
    manifest's ``joint_units`` is wrong and the runner will skip the deg->rad
    conversion that keeps the arm off its limits.
    """
    assert manifest.action_contract is not None
    assert manifest.action_contract.joint_units is not None
    assert manifest.action_contract.joint_units.value == "degrees"
    predicted, _ = rollout
    assert np.abs(predicted).max() > 10.0


def test_predicted_chunk_tracks_the_recorded_teleop(rollout: tuple[_Chunk, _Chunk]) -> None:
    """The load-bearing check: correct camera aliasing reproduces the demo."""
    predicted, truth = rollout
    mae = np.abs(predicted - truth).mean(axis=0)
    assert mae.max() < _MAE_CEILING, f"per-joint MAE {mae} exceeds {_MAE_CEILING}"


def test_predicted_chunk_actually_moves(rollout: tuple[_Chunk, _Chunk]) -> None:
    """Guard against the mean-collapse failure mode a badly-fed SmolVLA shows."""
    predicted, _ = rollout
    assert np.abs(predicted[-1] - predicted[0]).max() > 0.5


def test_starting_pose_is_inside_the_robot_joint_limits(manifest: RSkillManifest) -> None:
    """``starting_pose`` is clamped to so101 limits — verify it stayed there.

    The training start pose overshoots two of the SO-101 manifest limits (see
    the JOINT ENVELOPE note in rskill.yaml); the manifest ships the clamped
    pose so the pre-roll move is never rejected by the safety kernel.
    """
    from openral_core.schemas import RobotDescription

    robot = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"))
    pose = manifest.starting_pose
    assert pose is not None
    assert len(pose) == len(robot.joints)
    for value, joint in zip(pose, robot.joints, strict=True):
        limits = joint.position_limits
        assert limits is not None, f"{joint.name} declares no position_limits"
        low, high = limits
        assert low <= value <= high, f"{joint.name}={value} outside [{low}, {high}]"

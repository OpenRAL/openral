# SPDX-License-Identifier: Apache-2.0
"""The H2's S0 policy (NVIDIA WBC-AGILE, `nvidia/agile_wbc_unitree_h2`) rolls out as the
real InferenceController drives it.

`config/h2/policies/agile_velocity.yaml` points NVIDIA's C++ `InferenceController` at the
pinned Hub export. This drives that same ONNX the way the controller does — the
`pipeline.feedback_flow` outputs looped back to their inputs every tick, seeded from
`agile_velocity_initial_values.safetensors`, inputs mapped by `element_names` — for a
standing robot (default pose, upright, zero velocity command) and then a forward command,
and checks what reaches the hardware: 14 finite targets for exactly the policy's
`s0_joints`, within 1 rad of the default stand (a mis-ordered joint input or a wrong
scaling lands far outside), finite positive per-joint kp/kd; and that the velocity command
input is live.

What it does NOT show: balance, or targets inside the joint limits. The state is held
fixed (no physics), so the policy keeps stepping against a robot that never moves; in that
open loop it asks for a left ankle roll of -0.371 rad against H2.urdf's -0.349 rad limit
(measured 2026-11-24). Joint-limit clamping is the C++ chain's job on hardware; the
closed-loop MuJoCo S0 (balance, walking, clamping) is tests/sim/test_h2_agile_s0_mujoco.py.

Needs the export (NVIDIA Open Model License — downloaded, never vendored): the policy
file's `bundle_dir` or `$OPENRAL_H2_AGILE_DIR`. Skips with the download command otherwise;
a SHA-256 mismatch fails instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from openral_core import RobotDescription
from openral_hal._h2_agile_velocity import LeappVelocityPolicy
from openral_hal.unitree_s0_policy import load_s0_policy, resolve_leapp_config, s0_policy_path

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
pytest.importorskip("safetensors", reason="safetensors not installed")

_REPO = Path(__file__).resolve().parents[2]
_ENV = "OPENRAL_H2_AGILE_DIR"
_TICKS = 100  # 2 s at the export's 50 Hz


def _bundle() -> tuple[Path, dict[str, Any]]:
    policy = load_s0_policy(s0_policy_path("h2", "agile_velocity"))
    meta = policy.openral_policy
    override = os.environ.get(_ENV)
    leapp = (Path(override) if override else Path(str(meta.bundle_dir)).expanduser()) / (
        meta.leapp_config
    )
    if not leapp.is_file():
        pytest.skip(
            f"H2 AGILE export not downloaded — hf download {meta.hf_repo} --revision "
            f"{meta.hf_revision} --local-dir {meta.bundle_dir} (or set ${_ENV})"
        )
    if not override:
        resolve_leapp_config(policy, "")  # verifies every SHA-256 pin; mismatch fails
    return leapp, yaml.safe_load(leapp.read_text(encoding="utf-8"))


def _rollout(command: list[float]) -> list[dict[str, npt.NDArray[np.float32]]]:
    """Step the export (the twin's own runner) on a fixed standing state; every tick's outputs."""
    policy = LeappVelocityPolicy(_bundle()[0])
    q = np.array([policy.default_joint_pos[j] for j in policy.obs_joint_names])
    ticks = []
    for _ in range(_TICKS):
        policy.step(
            ang_vel_b=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            command=(command[0], command[1], command[2]),
            joint_pos=q,
            joint_vel=np.zeros(len(q)),
        )
        ticks.append(policy.last_outputs)
    return ticks


def _spec() -> dict[str, Any]:
    return _bundle()[1]


def test_standing_rollout_targets_only_the_s0_joints_near_the_stand() -> None:
    ticks = _rollout([0.0, 0.0, 0.0])
    spec = _spec()
    (model,) = spec["models"].values()
    outputs = {o["name"]: o for o in model["outputs"]}
    action_joints = outputs["joint_pos"]["element_names"][0]
    s0 = load_s0_policy(s0_policy_path("h2", "agile_velocity")).openral_policy.s0_joints
    assert sorted(action_joints) == sorted(s0)
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "h2" / "robot.yaml"))
    assert set(action_joints) <= {j.name for j in desc.joints}
    art = spec["agile"]["articulations"]["robot"]
    default = dict(zip(art["joint_names"], art["default_joint_pos"], strict=True))
    for tick in ticks:
        targets = tick["joint_pos"].reshape(-1)
        assert targets.shape == (14,) and np.isfinite(targets).all()
        for joint, value in zip(action_joints, targets, strict=True):
            assert abs(value - default[joint]) < 1.0, (joint, value)
        for gain in ("joint_pos_kp_gains", "joint_pos_kd_gains"):
            g = tick[gain].reshape(-1)
            assert g.shape == (14,) and np.isfinite(g).all() and (g > 0).all(), gain


def test_the_velocity_command_input_changes_the_leg_targets() -> None:
    """`command/body/velocity` is live: a forward command moves the targets off the stand."""
    still = _rollout([0.0, 0.0, 0.0])[-1]["joint_pos"]
    forward = _rollout([0.5, 0.0, 0.0])[-1]["joint_pos"]
    assert np.isfinite(forward).all()
    assert float(np.max(np.abs(forward - still))) > 1e-3

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the GR00T Decoupled WBC (Unitree G1) S0 policy as a LEAPP bundle for isaac_ros_deploy.

NVIDIA's ``InferenceController`` (isaac_ros_deploy v4.6-0 @ 3e84f85a) loads a *LEAPP* export: a YAML
naming one ONNX model's inputs/outputs by ``kind`` (``state/joint/position``,
``command/body/velocity``, ``target/joint/position``, ``kp``, …) plus a ``feedback_flow``
that loops recurrent outputs back to inputs every tick. GR00T-WholeBodyControl ships two
plain ONNX MLPs instead (``GR00T-WholeBodyControl-{Walk,Balance}.onnx``, input ``[B, 516]``,
output ``[B, 15]``) and leaves observation assembly, history, the walk/balance switch and
action scaling to a Python runner (``decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py``).

This tool moves that runner logic into ONE ONNX graph so the C++ controller can run it
unchanged — no new controller, no Python in the 200 Hz loop:

* per-tick observation (86): ``[cmd·(2, 2, 0.5), height 0.74, torso rpy cmd (0, 0, 0),
  pelvis angular velocity·0.5, projected gravity (from the IMU quaternion), q − q_default,
  dq·0.05, last action]`` — the runner's ``cmd_scale`` / ``height_cmd`` / ``rpy_cmd`` /
  ``ang_vel_scale`` / ``dof_*_scale`` from ``g1_gear_wbc.yaml``;
* 6-frame history, oldest first, zero-initialised: the previous 5 frames are a feedback
  tensor (``obs_history_in`` → ``obs_history_out``), as is ``last_action``;
* Walk if ``‖(vx, vy, wz)‖ > 0.05`` else Balance (both evaluated, selected with ``Where``);
* ``q* = 0.25·a + default`` for the 15 S0 joints (12 legs + waist yaw/roll/pitch), with the
  runner's per-joint ``kps`` / ``kds`` emitted as the ``kp`` / ``kd`` outputs.

The weights are NOT redistributed by OpenRAL (NVIDIA Open Model License; code Apache-2.0):
by default this downloads the two ONNX files from the pinned GR00T-WholeBodyControl commit
and refuses any whose SHA-256 differs, or takes local copies via ``--walk`` / ``--balance``
(same check). Parity with the original networks is ``tests/unit/test_gr00t_wbc_leapp_bundle.py``.

Usage::

    uv run python tools/build_gr00t_wbc_leapp.py            # → ~/.cache/openral/unitree/g1/...
    uv run python tools/build_gr00t_wbc_leapp.py --walk W.onnx --balance B.onnx --out DIR

The output directory is what ``packages/openral_hal_unitree/config/g1/policies/
gr00t_decoupled_wbc.yaml`` names as ``bundle_dir``; ``real_bringup.launch.py`` loads
``<bundle_dir>/gr00t_wbc_g1.yaml`` from there.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper, numpy_helper

#: GR00T-WholeBodyControl commit the weights and runner constants are pinned to.
GR00T_WBC_COMMIT = "1e1ec4f5f27e6c3e03048f693bd7c60cce04afda"
_POLICY_DIR = "decoupled_wbc/sim2mujoco/resources/robots/g1/policy"
#: (file name, SHA-256) per network — git-LFS objects of the pinned commit.
WALK_ONNX = (
    "GR00T-WholeBodyControl-Walk.onnx",
    "7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407",
)
BALANCE_ONNX = (
    "GR00T-WholeBodyControl-Balance.onnx",
    "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18",
)
_LFS_URL = "https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/{commit}/{path}"

DEFAULT_BUNDLE_DIR = Path("~/.cache/openral/unitree/g1/gr00t_decoupled_wbc").expanduser()
MODEL_NAME = "gr00t_wbc_g1"

#: The policy's 29-joint observation order (menagerie / unitree_mujoco / g1_gear_wbc.xml).
JOINTS29: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
#: Joints the policy's 15-D action commands (legs + all three waist axes).
S0_JOINTS: tuple[str, ...] = JOINTS29[:15]

# g1_gear_wbc.yaml @ GR00T_WBC_COMMIT.
DEFAULT_ANGLES = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2 + (0.0, 0.0, 0.0)
KPS = (150.0, 150.0, 150.0, 200.0, 40.0, 40.0) * 2 + (250.0, 250.0, 250.0)
KDS = (2.0, 2.0, 2.0, 4.0, 2.0, 2.0) * 2 + (5.0, 5.0, 5.0)
CMD_SCALE = (2.0, 2.0, 0.5)
HEIGHT_CMD = 0.74
RPY_CMD = (0.0, 0.0, 0.0)
ANG_VEL_SCALE = 0.5
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
OBS_DIM = 86
HISTORY = 6
WALK_THRESHOLD = 0.05
CONTROL_HZ = 50.0
_OPSET = 11  # the networks' own opset; the wrapper stays on it


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, expected: str) -> Path:
    """Return ``path`` if its SHA-256 is ``expected``.

    Raises:
        ValueError: On a mismatch — never build from weights that are not the pinned ones.
    """
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 {actual} is not the pinned {expected}")
    return path


def fetch(name_sha: tuple[str, str], dest_dir: Path) -> Path:
    """Download one pinned network from the GR00T-WholeBodyControl LFS store and verify it."""
    name, expected = name_sha
    dest = dest_dir / name
    if dest.is_file() and sha256(dest) == expected:
        return dest
    url = _LFS_URL.format(commit=GR00T_WBC_COMMIT, path=f"{_POLICY_DIR}/{name}")
    with (
        tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as tmp,
        urllib.request.urlopen(url, timeout=120) as response,  # reason: fixed https URL
    ):
        shutil.copyfileobj(response, tmp)
    verified(Path(tmp.name), expected)
    Path(tmp.name).replace(dest)
    return dest


def _const(
    name: str, values: Any, shape: tuple[int, ...] | None = None
) -> onnx.NodeProto:  # reason: array-like
    array = np.asarray(values, dtype=np.float32)
    if shape is not None:
        array = array.reshape(shape)
    return helper.make_node(
        "Constant", [], [name], value=numpy_helper.from_array(array, name=f"{name}_value")
    )


def _const_i64(name: str, values: list[int]) -> onnx.NodeProto:
    array = np.asarray(values, dtype=np.int64)
    return helper.make_node(
        "Constant", [], [name], value=numpy_helper.from_array(array, name=f"{name}_value")
    )


def _embedded(path: Path, prefix: str) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """The network's nodes + initializers, every name prefixed; I/O are ``<prefix>input/output``."""
    model = onnx.compose.add_prefix(onnx.load(str(path)), prefix=prefix)
    return list(model.graph.node), list(model.graph.initializer)


def build_model(walk: Path, balance: Path) -> onnx.ModelProto:
    """Compose the LEAPP-shaped wrapper graph around the two pinned networks."""
    f32 = TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info("velocity_command", f32, [1, 3]),
        helper.make_tensor_value_info("base_ang_vel", f32, [1, 3]),
        helper.make_tensor_value_info("base_quat", f32, [1, 4]),
        helper.make_tensor_value_info("joint_pos", f32, [1, 29]),
        helper.make_tensor_value_info("joint_vel", f32, [1, 29]),
        helper.make_tensor_value_info("obs_history_in", f32, [1, OBS_DIM * (HISTORY - 1)]),
        helper.make_tensor_value_info("last_action_in", f32, [1, 15]),
    ]
    outputs = [
        helper.make_tensor_value_info("joint_pos_target", f32, [1, 15]),
        helper.make_tensor_value_info("joint_pos_kp_gains", f32, [1, 15]),
        helper.make_tensor_value_info("joint_pos_kd_gains", f32, [1, 15]),
        helper.make_tensor_value_info("obs_history_out", f32, [1, OBS_DIM * (HISTORY - 1)]),
        helper.make_tensor_value_info("last_action_out", f32, [1, 15]),
    ]
    default29 = list(DEFAULT_ANGLES) + [0.0] * 14
    nodes: list[onnx.NodeProto] = [
        _const("c_cmd_scale", CMD_SCALE, (1, 3)),
        _const("c_height_rpy", (HEIGHT_CMD, *RPY_CMD), (1, 4)),
        _const("c_ang_vel_scale", ANG_VEL_SCALE),
        _const("c_default29", default29, (1, 29)),
        _const("c_dof_vel_scale", DOF_VEL_SCALE),
        _const("c_action_scale", ACTION_SCALE),
        _const("c_default15", DEFAULT_ANGLES, (1, 15)),
        _const("c_kp", KPS, (1, 15)),
        _const("c_kd", KDS, (1, 15)),
        _const("c_two", 2.0),
        _const("c_one", 1.0),
        _const("c_walk_threshold", WALK_THRESHOLD),
        helper.make_node("Mul", ["velocity_command", "c_cmd_scale"], ["o_cmd"]),
        helper.make_node("Mul", ["base_ang_vel", "c_ang_vel_scale"], ["o_ang_vel"]),
        helper.make_node("Sub", ["joint_pos", "c_default29"], ["o_q"]),
        helper.make_node("Mul", ["joint_vel", "c_dof_vel_scale"], ["o_dq"]),
    ]
    # Projected gravity R^T·[0,0,-1] from (qx, qy, qz, qw) — the runner's
    # get_gravity_orientation: [2(wy − zx), −2(zy + wx), 1 − 2(w² + z²)].
    for i, c in enumerate("xyzw"):
        nodes += [
            _const_i64(f"i_{c}", [i]),
            helper.make_node("Gather", ["base_quat", f"i_{c}"], [f"q{c}"], axis=1),
        ]
    nodes += [
        helper.make_node("Mul", ["qw", "qy"], ["wy"]),
        helper.make_node("Mul", ["qz", "qx"], ["zx"]),
        helper.make_node("Sub", ["wy", "zx"], ["g0a"]),
        helper.make_node("Mul", ["g0a", "c_two"], ["g0"]),
        helper.make_node("Mul", ["qz", "qy"], ["zy"]),
        helper.make_node("Mul", ["qw", "qx"], ["wx"]),
        helper.make_node("Add", ["zy", "wx"], ["g1a"]),
        helper.make_node("Mul", ["g1a", "c_two"], ["g1b"]),
        helper.make_node("Neg", ["g1b"], ["g1"]),
        helper.make_node("Mul", ["qw", "qw"], ["ww"]),
        helper.make_node("Mul", ["qz", "qz"], ["zz"]),
        helper.make_node("Add", ["ww", "zz"], ["g2a"]),
        helper.make_node("Mul", ["g2a", "c_two"], ["g2b"]),
        helper.make_node("Sub", ["c_one", "g2b"], ["g2"]),
        helper.make_node("Concat", ["g0", "g1", "g2"], ["o_gravity"], axis=1),
        helper.make_node(
            "Concat",
            [
                "o_cmd",
                "c_height_rpy",
                "o_ang_vel",
                "o_gravity",
                "o_q",
                "o_dq",
                "last_action_in",
            ],
            ["obs"],
            axis=1,
        ),
        # History: previous 5 frames (oldest first) then this frame → [1, 516].
        helper.make_node("Concat", ["obs_history_in", "obs"], ["x"], axis=1),
        _const_i64("i_hist", list(range(OBS_DIM, OBS_DIM * HISTORY))),
        helper.make_node("Gather", ["x", "i_hist"], ["obs_history_out"], axis=1),
        helper.make_node("Identity", ["x"], ["walk/input"]),
        helper.make_node("Identity", ["x"], ["balance/input"]),
    ]
    walk_nodes, walk_init = _embedded(walk, "walk/")
    bal_nodes, bal_init = _embedded(balance, "balance/")
    nodes += walk_nodes + bal_nodes
    nodes += [
        # Walk iff ‖(vx, vy, wz)‖ > 0.05 on the UNscaled command (runner: np.linalg.norm).
        helper.make_node("ReduceL2", ["velocity_command"], ["cmd_norm"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["cmd_norm", "c_walk_threshold"], ["use_walk"]),
        helper.make_node("Where", ["use_walk", "walk/output", "balance/output"], ["action"]),
        helper.make_node("Identity", ["action"], ["last_action_out"]),
        helper.make_node("Mul", ["action", "c_action_scale"], ["scaled_action"]),
        helper.make_node("Add", ["scaled_action", "c_default15"], ["joint_pos_target"]),
        helper.make_node("Identity", ["c_kp"], ["joint_pos_kp_gains"]),
        helper.make_node("Identity", ["c_kd"], ["joint_pos_kd_gains"]),
    ]
    graph = helper.make_graph(
        nodes, MODEL_NAME, inputs, outputs, initializer=[*walk_init, *bal_init]
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", _OPSET)],
        producer_name="openral tools/build_gr00t_wbc_leapp.py",
    )
    model.ir_version = 6
    onnx.checker.check_model(model, full_check=True)
    return model


def leapp_config(onnx_name: str, onnx_sha256: str) -> dict[str, Any]:
    """The LEAPP YAML (format of isaac_ros_deploy's ``parse_graph_config``)."""

    def term(name: str, shape: list[int], **extra: Any) -> dict[str, Any]:
        return {"name": name, "dtype": "float32", "shape": shape, "type": "tensor", **extra}

    m = MODEL_NAME
    return {
        "models": {
            m: {
                "inputs": [
                    term(
                        "velocity_command",
                        [1, 3],
                        kind="command/body/velocity",
                        element_names=[["lin_vel_x", "lin_vel_y", "ang_vel_z"]],
                    ),
                    term(
                        "base_ang_vel",
                        [1, 3],
                        kind="state/body/angular_velocity",
                        element_names=[["x", "y", "z"]],
                    ),
                    term(
                        "base_quat",
                        [1, 4],
                        kind="state/body/rotation",
                        element_names=[["qx", "qy", "qz", "qw"]],
                    ),
                    term(
                        "joint_pos",
                        [1, 29],
                        kind="state/joint/position",
                        element_names=[list(JOINTS29)],
                    ),
                    term(
                        "joint_vel",
                        [1, 29],
                        kind="state/joint/velocity",
                        element_names=[list(JOINTS29)],
                    ),
                    term("obs_history_in", [1, OBS_DIM * (HISTORY - 1)]),
                    term("last_action_in", [1, 15]),
                ],
                "outputs": [
                    term(
                        "joint_pos_target",
                        [1, 15],
                        kind="target/joint/position",
                        element_names=[list(S0_JOINTS)],
                    ),
                    term("joint_pos_kp_gains", [1, 15], kind="kp", element_names=[list(S0_JOINTS)]),
                    term("joint_pos_kd_gains", [1, 15], kind="kd", element_names=[list(S0_JOINTS)]),
                    term("obs_history_out", [1, OBS_DIM * (HISTORY - 1)]),
                    term("last_action_out", [1, 15]),
                ],
                "parameters": {
                    "model_path": onnx_name,
                    "sha256sum": onnx_sha256,
                    "backend": "onnx",
                    "opset_version": _OPSET,
                },
            }
        },
        "pipeline": {
            "inputs": {
                m: ["velocity_command", "base_ang_vel", "base_quat", "joint_pos", "joint_vel"]
            },
            "outputs": {m: ["joint_pos_target", "joint_pos_kp_gains", "joint_pos_kd_gains"]},
            "data_flow": {},
            # Zero-initialised (no initial_values): the runner's zero history / zero action.
            "feedback_flow": {
                f"{m}/obs_history_out": [f"{m}/obs_history_in"],
                f"{m}/last_action_out": [f"{m}/last_action_in"],
            },
            "configs": {"frequency": CONTROL_HZ},
        },
        "openral": {
            "source": f"NVlabs/GR00T-WholeBodyControl@{GR00T_WBC_COMMIT}",
            "walk_sha256": WALK_ONNX[1],
            "balance_sha256": BALANCE_ONNX[1],
            "weights_license": "NVIDIA Open Model License",
        },
    }


def build_bundle(walk: Path, balance: Path, out_dir: Path) -> Path:
    """Verify both networks, write ``<out_dir>/gr00t_wbc_g1.{onnx,yaml}``, return the YAML."""
    verified(walk, WALK_ONNX[1])
    verified(balance, BALANCE_ONNX[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{MODEL_NAME}.onnx"
    onnx.save(build_model(walk, balance), str(onnx_path))
    yaml_path = out_dir / f"{MODEL_NAME}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(leapp_config(onnx_path.name, sha256(onnx_path)), sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--walk", type=Path, help="local GR00T-WholeBodyControl-Walk.onnx")
    parser.add_argument("--balance", type=Path, help="local GR00T-WholeBodyControl-Balance.onnx")
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_DIR)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    walk = args.walk or fetch(WALK_ONNX, args.out)
    balance = args.balance or fetch(BALANCE_ONNX, args.out)
    print(build_bundle(walk, balance, args.out))  # reason: CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

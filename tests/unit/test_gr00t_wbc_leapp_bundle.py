# SPDX-License-Identifier: Apache-2.0
"""The GR00T WBC LEAPP bundle must compute what GR00T's own runner computes.

`tools/build_gr00t_wbc_leapp.py` folds the GR00T-WholeBodyControl Python runner
(observation scaling, projected gravity, 6-frame history, last action, the walk/balance
switch, ``0.25·a + default``) into one ONNX graph so NVIDIA's C++ ``InferenceController``
can run the policy as the G1's real-hardware S0. That is only acceptable if the bundle,
driven exactly as the controller drives it (feedback outputs looped back to inputs every
tick, zero-initialised), matches the runner on the ORIGINAL pinned networks.

The reference below is an independent port of ``run_mujoco_gear_wbc.py`` @ 1e1ec4f5
(`GearWbcController.obs` / `act`), run on the two original ONNX files through onnxruntime.
Both are fed the same seeded 120-tick input sequence, which crosses the 0.05 walk threshold
in both directions, and must agree on every tick's joint targets.

Needs the two pinned networks (NVIDIA Open Model License — not in this repo):
`$OPENRAL_GR00T_WBC_ONNX_DIR` (holding ``GR00T-WholeBodyControl-{Walk,Balance}.onnx``) or
the bundle cache `tools/build_gr00t_wbc_leapp.py` downloads into. Skips with the reason
otherwise; a SHA-256 mismatch fails instead.
"""

from __future__ import annotations

import collections
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "tools" / "build_gr00t_wbc_leapp.py"
_F64 = npt.NDArray[np.float64]

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
pytest.importorskip("onnx", reason="onnx not installed")


def _tool() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("build_gr00t_wbc_leapp", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _networks(tool: types.ModuleType) -> tuple[Path, Path]:
    candidates = [Path(p) for p in (os.environ.get("OPENRAL_GR00T_WBC_ONNX_DIR"),) if p]
    candidates.append(tool.DEFAULT_BUNDLE_DIR)
    for d in candidates:
        walk, bal = d / tool.WALK_ONNX[0], d / tool.BALANCE_ONNX[0]
        if walk.is_file() and bal.is_file():
            return walk, bal
    pytest.skip(
        "GR00T WBC networks not found — set $OPENRAL_GR00T_WBC_ONNX_DIR or run "
        "tools/build_gr00t_wbc_leapp.py (downloads the pinned NVIDIA Open Model License weights)"
    )


class _ReferenceRunner:
    """Independent port of GR00T's GearWbcController (run_mujoco_gear_wbc.py @ 1e1ec4f5)."""

    kps_yaml = (150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250.0, 250.0, 250.0)
    default15 = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0], np.float64)
    cmd_scale = np.array([2.0, 2.0, 0.5], np.float32)

    def __init__(self, walk: Path, balance: Path) -> None:
        self.walk = ort.InferenceSession(str(walk), providers=["CPUExecutionProvider"])
        self.balance = ort.InferenceSession(str(balance), providers=["CPUExecutionProvider"])
        self.action = np.zeros(15, np.float32)
        self.hist = collections.deque([np.zeros(86, np.float32)] * 6, maxlen=6)
        self.default29 = np.zeros(29)
        self.default29[:15] = self.default15

    def act(self, cmd: _F64, ang_vel: _F64, quat_wxyz: _F64, q: _F64, dq: _F64) -> _F64:
        o = np.zeros(86, np.float32)
        o[0:3] = cmd.astype(np.float32) * self.cmd_scale
        o[3] = 0.74
        o[4:7] = 0.0
        o[7:10] = ang_vel * 0.5
        w, x, y, z = quat_wxyz
        o[10:13] = [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)]
        o[13:42] = q - self.default29
        o[42:71] = dq * 0.05
        o[71:86] = self.action
        self.hist.append(o)
        x_in = np.concatenate(list(self.hist))[None].astype(np.float32)
        sess = self.walk if np.linalg.norm(cmd) > 0.05 else self.balance
        self.action = sess.run(None, {"input": x_in})[0][0].astype(np.float32)
        return np.asarray(self.action * 0.25 + self.default15, dtype=np.float64)


def _inputs(n: int) -> list[dict[str, _F64]]:
    """Seeded tick inputs; the command steps through balance → walk → balance → walk."""
    rng = np.random.default_rng(20261107)
    default29 = np.zeros(29)
    default29[:15] = _ReferenceRunner.default15
    cmds = [np.zeros(3), np.array([0.02, 0.0, 0.01]), np.array([0.4, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]), np.array([0.2, -0.15, 0.3])]  # fmt: skip
    ticks = []
    for i in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.0, 0.15)
        quat_wxyz = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
        ticks.append(
            {
                "cmd": cmds[(i // 24) % len(cmds)] + rng.normal(scale=0.005, size=3),
                "ang_vel": rng.normal(scale=0.3, size=3),
                "quat_wxyz": quat_wxyz,
                "q": default29 + rng.normal(scale=0.08, size=29),
                "dq": rng.normal(scale=1.0, size=29),
            }
        )
    return ticks


def test_the_bundle_matches_the_original_runner_tick_for_tick(tmp_path: Path) -> None:
    tool = _tool()
    walk, balance = _networks(tool)
    leapp_yaml = tool.build_bundle(walk, balance, tmp_path)
    leapp = yaml.safe_load(leapp_yaml.read_text())
    model = leapp["models"][tool.MODEL_NAME]
    bundle = ort.InferenceSession(
        str(tmp_path / model["parameters"]["model_path"]), providers=["CPUExecutionProvider"]
    )
    reference = _ReferenceRunner(walk, balance)
    # Feedback state exactly as InferenceController keeps it: zeros, then each tick's outputs.
    feedback = {
        "obs_history_in": np.zeros((1, 430), np.float32),
        "last_action_in": np.zeros((1, 15), np.float32),
    }
    worst = 0.0
    walked = balanced = 0
    for tick in _inputs(120):
        w, x, y, z = tick["quat_wxyz"]
        outs = dict(
            zip(
                [o.name for o in bundle.get_outputs()],
                bundle.run(
                    None,
                    {
                        "velocity_command": tick["cmd"][None].astype(np.float32),
                        "base_ang_vel": tick["ang_vel"][None].astype(np.float32),
                        "base_quat": np.array([[x, y, z, w]], np.float32),
                        "joint_pos": tick["q"][None].astype(np.float32),
                        "joint_vel": tick["dq"][None].astype(np.float32),
                        **feedback,
                    },
                ),
                strict=True,
            )
        )
        feedback = {
            "obs_history_in": outs["obs_history_out"],
            "last_action_in": outs["last_action_out"],
        }
        expected = reference.act(
            tick["cmd"], tick["ang_vel"], tick["quat_wxyz"], tick["q"], tick["dq"]
        )
        if np.linalg.norm(tick["cmd"]) > 0.05:
            walked += 1
        else:
            balanced += 1
        worst = max(worst, float(np.max(np.abs(outs["joint_pos_target"][0] - expected))))
        np.testing.assert_allclose(outs["joint_pos_target"][0], expected, atol=1e-5)
        np.testing.assert_allclose(outs["joint_pos_kp_gains"][0], reference.kps_yaml)
    assert walked > 20 and balanced > 20, (walked, balanced)
    assert worst < 1e-5


def test_the_leapp_yaml_has_what_isaac_ros_deploys_parser_requires(tmp_path: Path) -> None:
    """`parse_graph_config` / `merge_graph_to_model_config` contract (isaac_ros_deploy v4.6-0)."""
    tool = _tool()
    walk, balance = _networks(tool)
    doc = yaml.safe_load(tool.build_bundle(walk, balance, tmp_path).read_text())
    (name,) = doc["models"]
    model = doc["models"][name]
    assert model["parameters"]["backend"] == "onnx"  # the only backend the controller runs
    assert (tmp_path / model["parameters"]["model_path"]).is_file()
    for term in model["inputs"] + model["outputs"]:
        assert term["name"] and term["shape"]
    out_names = {o["name"] for o in model["outputs"]}
    for key, targets in doc["pipeline"]["feedback_flow"].items():
        model_prefix, output = key.split("/", 1)
        assert model_prefix == name and output in out_names
        assert all(t.startswith(f"{name}/") for t in targets)
    # The S0 action covers exactly the joints the policy file gives the policy.
    policy = yaml.safe_load(
        (
            _REPO / "packages/openral_hal_unitree/config/g1/policies/gr00t_decoupled_wbc.yaml"
        ).read_text()
    )
    (target,) = [o for o in model["outputs"] if o.get("kind") == "target/joint/position"]
    assert target["element_names"][0] == policy["openral_policy"]["s0_joints"]

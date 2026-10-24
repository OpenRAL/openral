"""Issue #373 collision tier: the real kernel stops the MuJoCo OpenArm on contact, not before.

The OpenArm v2 MJCF runs in this process and publishes ``/joint_states`` with the
effort a MuJoCo HAL publishes (``qfrc_actuator``, #388) to the real
``safety_kernel_node``, configured exactly as ``deploy_e2e.launch.py`` would be from
``robots/openarm/robot.yaml`` (its lowered ``force_observer`` model) plus a
deploy-scene ceiling of ``_CEILING_NM`` on the right arm's joints 1-4:

==========================================================  ====================================
row                                                         verdict
==========================================================  ====================================
right arm on a sinusoid, then a 40 N push down on the hand  no stop while free; ``KIND_FORCE``
over ``[_PUSH_S, _PUSH_S + 0.5)``                           within ``_DETECT_S`` of the push
right arm swung into a box fixed in the world               no stop before first contact;
                                                            ``KIND_FORCE`` within ``_DETECT_S``
==========================================================  ====================================

The observer's model is the plant's own (lowered from the same MJCF), so this
proves the wiring and the dynamics, not a real-arm floor: the real residual
carries stiction and offsets the sim has none of
(``docs/reference/manipulation-workspace-contact.md`` §8).

Gates: ``OPENRAL_TEST_ROS_LIVE=1`` + ROS_DISTRO + rclpy + openral_msgs + mujoco +
colcon-built kernel. ``scripts/ros_live_tests.sh`` runs it.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
import uuid
from collections.abc import Callable
from typing import Any

import pytest

_LIVE = os.environ.get("OPENRAL_TEST_ROS_LIVE") == "1" and bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="live-ROS test — set OPENRAL_TEST_ROS_LIVE=1 with ROS 2 sourced "
    "(scripts/ros_live_tests.sh does both).",
)
if _LIVE:
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed (sim extra)")

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from openral_core import RobotDescription, SafetyEnvelope  # noqa: E402
from openral_core.assets import resolve_asset  # noqa: E402
from openral_safety.envelope_loader import (  # noqa: E402
    collision_params_from_description,
    compute_intersection,
    kernel_params_from_envelope,
)

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"
_CEILING_NM = 3.0
_MONITORED = ("right_joint1", "right_joint2", "right_joint3", "right_joint4")
_PUBLISH_HZ = 250.0
_PUSH_S = 2.0
_PUSH_N = 40.0
_DETECT_S = 0.15
_Q0 = {"right_joint2": 0.3, "right_joint4": 1.2, "left_joint4": 1.2}


def _robot() -> RobotDescription:
    return RobotDescription.from_yaml(str(_MANIFEST))


def _kernel_params(robot: RobotDescription) -> dict[str, object]:
    deploy = SafetyEnvelope(max_external_joint_torque_nm=dict.fromkeys(_MONITORED, _CEILING_NM))
    params: dict[str, object] = {
        **kernel_params_from_envelope(compute_intersection(robot, None, deploy=deploy)),
        **collision_params_from_description(robot),
    }
    params["collision_joint_names"] = [j.name for j in robot.joints]
    params["collision_state_deadline_ms"] = 200.0
    return params


class _Plant:
    """The OpenArm MJCF under its own position actuators, read as a HAL publishes it."""

    def __init__(self, robot: RobotDescription, model: Any) -> None:
        self.m = model
        self.d = mujoco.MjData(model)
        self.names = [j.name for j in robot.joints]
        joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.sim_joint_name or j.name)
            for j in robot.joints
        ]
        assert min(joints) >= 0, "every manifest joint has an MJCF joint"
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dofs = np.array([model.jnt_dofadr[j] for j in joints])
        # Each manifest joint's actuator drives its MJCF joint directly.
        self.act = {
            int(model.actuator_trnid[a, 0]): a
            for a in range(model.nu)
            if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        }
        self.hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_ee_base_link")
        self.set_target(_Q0)
        self.d.qpos[self.qadr] = [_Q0.get(n, 0.0) for n in self.names]
        mujoco.mj_forward(model, self.d)

    def set_target(self, q: dict[str, float]) -> None:
        for name, joint in zip(self.names, self.qadr, strict=True):
            jid = int(np.flatnonzero(self.m.jnt_qposadr == joint)[0])
            if jid in self.act:
                self.d.ctrl[self.act[jid]] = q.get(name, 0.0)

    def step(self, seconds: float) -> None:
        for _ in range(round(seconds / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)  # torque and state as one sample

    def sample(self) -> tuple[list[float], list[float], list[float]]:
        return (
            [float(v) for v in self.d.qpos[self.qadr]],
            [float(v) for v in self.d.qvel[self.dofs]],
            [float(v) for v in self.d.qfrc_actuator[self.dofs]],
        )

    def external_torque(self, force: NDArray[np.float64]) -> dict[str, float]:
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, self.hand)
        tau = jac.T @ force
        return {n: float(tau[d]) for n, d in zip(self.names, self.dofs, strict=True)}


def _run(
    model: Any,
    seconds: float,
    drive: Callable[[_Plant, float], None],
) -> tuple[list[Any], dict[str, float], _Plant, list[tuple[float, list[float]]], str]:
    """Run the plant through the real kernel; returns (force triggers, sim times, plant, log)."""
    import rclpy
    from openral_msgs.msg import FailureTrigger
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    robot = _robot()
    plant = _Plant(robot, model)
    node_name = f"safety_kernel_force_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            _kernel_params(robot),
            node_name,
            isolated_domain_id(),
            log_path=log_path,
            params_file=pathlib.Path(td) / "kernel_params.yaml",
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("force_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"
                triggers: list[Any] = []
                times: dict[str, float] = {}
                estimates: list[tuple[float, list[float]]] = []
                now = {"t": 0.0}

                def on_failure(msg: Any) -> None:
                    if msg.kind == FailureTrigger.KIND_FORCE:
                        times.setdefault("trip", now["t"])
                        triggers.append(msg)

                helper.create_subscription(
                    FailureTrigger, "/openral/failure/safety", on_failure, 10
                )
                helper.create_subscription(
                    JointState,
                    "/openral/safety/external_torque",
                    lambda m: estimates.append((now["t"], list(m.effort))),
                    qos_profile_sensor_data,
                )
                pub = helper.create_publisher(JointState, "/joint_states", 10)
                deadline = time.time() + 5.0
                while time.time() < deadline and pub.get_subscription_count() < 1:
                    rclpy.spin_once(helper, timeout_sec=0.05)
                period = 1.0 / _PUBLISH_HZ
                base_ns = helper.get_clock().now().nanoseconds
                wall0 = time.monotonic()
                k = 0
                while k * period < seconds:
                    t = k * period
                    now["t"] = t
                    drive(plant, t)
                    plant.step(period)
                    q, qd, tau = plant.sample()
                    js = JointState()
                    stamp = base_ns + round((t + period) * 1e9)
                    js.header.stamp.sec, js.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
                    js.name = plant.names
                    js.position, js.velocity, js.effort = q, qd, tau
                    pub.publish(js)
                    k += 1
                    while time.monotonic() - wall0 < k * period:
                        rclpy.spin_once(helper, timeout_sec=0.0005)
                end = time.time() + 0.5
                while time.time() < end:
                    rclpy.spin_once(helper, timeout_sec=0.02)
            finally:
                rclpy.shutdown()
        finally:
            terminate_kernel(proc)
        log = log_path.read_text(encoding="utf-8", errors="replace")
    return triggers, times, plant, estimates, log


def _peak_before(
    estimates: list[tuple[float, list[float]]], names: list[str], t_end: float
) -> float:
    cols = [names.index(n) for n in _MONITORED]
    return max(
        (abs(e[c]) for t, e in estimates if 0.3 <= t < t_end for c in cols if c < len(e)),
        default=0.0,
    )


def _model(robot: RobotDescription, box_at: NDArray[np.float64] | None = None) -> Any:
    assert robot.assets.mjcf is not None
    path = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=_MANIFEST.parent)
    assert path is not None
    if box_at is None:
        return mujoco.MjModel.from_xml_path(str(path))
    spec = mujoco.MjSpec.from_file(str(path))
    body = spec.worldbody.add_body(name="obstacle", pos=box_at.tolist())
    body.add_geom(name="obstacle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.12, 0.04])
    return spec.compile()


def test_a_push_on_the_moving_hand_stops_the_arm_and_free_motion_does_not() -> None:
    robot = _robot()
    truth: dict[str, float] = {}

    def drive(plant: _Plant, t: float) -> None:
        q = dict(_Q0)
        q["right_joint1"] = 0.4 * np.sin(2 * np.pi * 0.3 * t)
        q["right_joint3"] = 0.4 * np.sin(2 * np.pi * 0.35 * t)
        q["right_joint4"] = 1.2 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
        plant.set_target(q)
        on = _PUSH_S <= t < _PUSH_S + 0.5
        plant.d.xfrc_applied[plant.hand, :3] = (0.0, 0.0, -_PUSH_N) if on else (0.0, 0.0, 0.0)
        if on and not truth:
            truth.update(plant.external_torque(np.array([0.0, 0.0, -_PUSH_N])))

    triggers, times, plant, estimates, log = _run(_model(robot), _PUSH_S + 1.0, drive)
    assert max(abs(truth[n]) for n in _MONITORED) > 1.5 * _CEILING_NM, (
        f"precondition: the push must load a monitored joint well past the ceiling: {truth}"
    )
    assert estimates, f"the kernel published no estimate\n{log}"
    peak = _peak_before(estimates, plant.names, _PUSH_S)
    assert peak < _CEILING_NM, f"free motion read {peak:.2f} N·m of 'contact'\n{log}"
    assert triggers, f"the push was not stopped\n{log}"
    assert _PUSH_S <= times["trip"] <= _PUSH_S + _DETECT_S, (times, log)
    evidence = json.loads(triggers[0].evidence_json)
    assert evidence["unit"] == "Nm" and evidence["joint_or_ee"] in _MONITORED, evidence
    print(
        f"push: free-motion peak {peak:.3f} N·m, stop {times['trip'] - _PUSH_S:.3f} s after onset"
    )


def test_the_arm_swung_into_a_fixed_box_is_stopped_at_contact_not_before() -> None:
    robot = _robot()
    probe = _Plant(robot, _model(robot))
    probe.d.qpos[probe.qadr[probe.names.index("right_joint1")]] = 0.5
    mujoco.mj_forward(probe.m, probe.d)
    box_at = np.array(probe.d.xpos[probe.hand]) + np.array([0.0, 0.0, 0.06])
    model = _model(robot, box_at)
    obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
    contact: dict[str, float] = {}

    def drive(plant: _Plant, t: float) -> None:
        q = dict(_Q0)
        q["right_joint1"] = min(0.8, 0.8 * max(0.0, t - 0.5) / 1.5)
        plant.set_target(q)
        hit = any(
            obstacle in (plant.d.contact[i].geom1, plant.d.contact[i].geom2)
            for i in range(plant.d.ncon)
        )
        if hit:
            contact.setdefault("t", t)

    triggers, times, plant, estimates, log = _run(model, 2.5, drive)
    assert "t" in contact, "precondition: the swing must reach the box"
    assert triggers, f"the collision was not stopped\n{log}"
    assert times["trip"] >= contact["t"], ("stopped before contact", times, contact, log)
    assert times["trip"] <= contact["t"] + _DETECT_S, (times, contact, log)
    peak = _peak_before(estimates, plant.names, contact["t"])
    print(
        f"collision: pre-contact peak {peak:.3f} N·m, stop {times['trip'] - contact['t']:.3f} s "
        "after first contact"
    )

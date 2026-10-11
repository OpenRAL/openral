"""Sim test: the Isaac sidecar's ``contacts`` endpoint reports a real PhysX penetration.

Boots ``scenes/deploy/isaac_franka_urdf.yaml`` with one extra **static** box
(a USDA cube with a collider, no rigid body) placed through the Panda's
``panda_link1``, so the robot starts in contact with world geometry. Asserts that
``_IsaacSimSidecar.contact_ground_truth`` -- the call ``SimSensorBridge`` makes on
every kernel stop -- reports that robot-link/box pair with ``min_separation_m <= 0``,
and that ``isaac_estop_ground_truth_record`` turns the reply into a JSON-safe
``sim.estop_ground_truth_snapshot`` body carrying a positive ``max_penetration_m``.

Skip policy: as ``test_franka_urdf_isaac`` -- needs pyzmq/msgpack on the openral
venv and a provisioned Isaac sidecar (RTX GPU); CI without them skips (§1.12).
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.sim.conftest import _repo_root, _sidecar_python_available

_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="isaac_sim wire needs " + ", ".join(_WIRE_MISSING) + " (just sync --group isaacsim)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason="Isaac Sim sidecar venv not provisioned (set OPENRAL_ISAAC_SIDECAR_PYTHON)",
    ),
]

# A 10 cm static cube 10 cm in front of the base axis at link1 height: its near
# face (x = 0.05 m) is inside panda_link1's ~6 cm-radius housing. The cube is a
# CHILD of the default prim: the scene references an object under a prim it
# defines as an Xform, which would retype a root Cube and drop its geometry.
_BOX_USDA = """#usda 1.0
(
    defaultPrim = "contact_box"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "contact_box"
{
    def Cube "geom" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 0.1
        float3[] extent = [(-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)]
    }
}
"""
_BOX_XYZ = [0.10, 0.0, 0.25]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    tmp: Path = tmp_path_factory.mktemp("isaac_contact")
    box = tmp / "contact_box.usda"
    box.write_text(_BOX_USDA, encoding="utf-8")
    cfg = yaml.safe_load(
        (_repo_root() / "scenes" / "deploy" / "isaac_franka_urdf.yaml").read_text(encoding="utf-8")
    )
    cfg["scene"]["backend_options"]["objects"] = [
        {"name": "contact_box", "usd": str(box), "xyz": _BOX_XYZ, "dynamic": False}
    ]
    scene_yaml = tmp / "isaac_franka_contact.yaml"
    scene_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    sim_env, seed = build_sim_env_from_yaml(str(scene_yaml), robot_id_fallback="franka_panda")
    sim_env.reset(seed=seed)
    yield sim_env
    sim_env.close()


def test_contacts_report_the_robot_link_inside_the_static_box(env: Any) -> None:
    from openral_hal.sim_sensor_bridge import isaac_estop_ground_truth_record

    for _ in range(3):
        env.step(np.full(env.action_dim, np.nan, dtype=np.float32))  # hold
    reply = env.contact_ground_truth()
    print("contacts reply:", json.dumps(reply, sort_keys=True))
    box_pairs = [
        p
        for p in reply["robot_world_contacts"]
        if p["other_prim"].startswith("/World/objects/contact_box")
    ]
    assert box_pairs, f"no robot/box pair in {reply['robot_world_contacts']}"
    deepest = box_pairs[0]
    assert deepest["min_separation_m"] <= 0.0, deepest
    assert deepest["contact_points"] >= 1
    assert deepest["robot_collider"].startswith(reply["robot_prim"].rstrip("/") + "/")
    assert set(deepest["robot_collider_offsets"]) == {"contact_offset_m", "rest_offset_m"}
    assert reply["nearest_robot_world_pairs"] is None
    assert reply["nearest_unavailable_reason"]
    assert reply["sim_time_ns"] > 0

    record = isaac_estop_ground_truth_record(reply)
    line = "sim.estop_ground_truth_snapshot " + json.dumps(
        {"stop_seq": 1, **record}, sort_keys=True
    )
    print(line)
    assert record["backend"] == "isaacsim"
    assert record["max_penetration_m"] >= -deepest["min_separation_m"]
    assert "contactOffset" in str(record["contacts_caveat"])

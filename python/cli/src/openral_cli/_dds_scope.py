# SPDX-License-Identifier: Apache-2.0
"""Keep a simulation and a real robot off each other's ROS graph.

**The incident this exists for (2026-09-05, issue #227).** `openral deploy sim`
set no DDS scope. It inherited the environment, and `ROS_DOMAIN_ID` is normally
unset — domain 0, with multicast discovery across the whole subnet. A simulation
launched on one host joined the ROS graph of a *live bimanual OpenArm* running
on another host on the same LAN. `/joint_states` had two publishers; the sim's
state assembler read `openarm_left_joint1 … openarm_right_joint7` where it
wanted `panda_gripper`, and every round of a 10-round A/B died in ~50 s looking
exactly like a policy failure.

Nothing actuated, and that was luck rather than design: the robot's stack
consumed its own topic names and happened to have no subscriber on
`/openral/candidate_action`. A simulation publishing an `Action` onto a topic a
physical robot subscribes to is one name collision away.

Two independent controls, because they fail differently:

* :func:`confine_sim_scope` — *prevention*. A sim graph lives on one host, so it
  is pinned to that host and to a private domain. Measured against the live
  OpenArm: with `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET` (the default) the robot's
  nodes are visible; with `LOCALHOST` they are not, while same-host discovery is
  unaffected. `OFF` would also hide the sim's own nodes from each other.
* :func:`assert_graph_unoccupied` — *detection*, and it is the half that covers
  the direction confinement cannot. Confinement stops a sim reaching a robot; it
  does nothing about a real-robot launch joining a graph a sim is already on, and
  nothing about two runs on one host colliding inside the same private domain.

**The signature is a `/joint_states` publisher, deliberately.** Every robot has
exactly one — real or simulated — so one rule covers both directions without
either side having to recognise the other's node names. If one is already there
when you launch, you are about to share a graph with another robot.

.. warning::
   The ``ros2`` CLI daemon is **not** usable for this check. It is a long-lived
   process that answers from its own environment, so ``ros2 node list`` under
   ``ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`` still reported the remote robot's
   nodes — a false negative that reads as "the setting does not work". The probe
   here runs in a subprocess with the exact launch environment and talks to
   ``rclpy`` directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Final

from openral_core.exceptions import ROSConfigError

__all__ = [
    "ALLOW_SHARED_GRAPH_ENV",
    "SIM_DOMAIN_ID",
    "assert_graph_unoccupied",
    "confine_sim_scope",
]

#: Escape hatch, matching the repo's other "I know what I am doing" env gates
#: (``OPENRAL_ALLOW_REMOTE_CODE``, ``OPENRAL_ALLOW_UNSAFE_PICKLE``). Set it to
#: attach a dashboard from another host, or to run a sim beside a robot on
#: purpose. Named in every refusal below so the way out is never a guess.
ALLOW_SHARED_GRAPH_ENV: Final[str] = "OPENRAL_ALLOW_SHARED_GRAPH"

#: The domain a confined sim runs on. Any fixed value would do — what matters is
#: that it is not 0, which is where an unconfigured robot host lands.
SIM_DOMAIN_ID: Final[str] = "77"

#: How long the probe listens before deciding the graph is empty. DDS discovery
#: is asynchronous: a participant that exists is not necessarily *known* the
#: instant a new one joins. Too short and the guard passes on a graph that is
#: occupied, which is the failure direction that matters.
_PROBE_SPIN_S: Final[float] = 3.0

_PROBE_NODE_NAME: Final[str] = "openral_graph_scope_probe"

#: Returned by :func:`_scan_graph` when ``rclpy`` is not importable at all.
#: Distinct from "the probe failed": a host with no ROS has no graph to join and
#: cannot run ``ros2 launch`` either, so the launch fails on its own with a
#: clearer message than this guard could give. Refusing there would block every
#: non-ROS machine — including CI — from exercising the launch path, and would
#: buy no safety, because there is no robot to collide with.
_NO_ROS: Final[str] = "no-ros"

#: How many occupying node names the refusal lists before eliding. Enough to
#: recognise a ros2_control stack at a glance without burying the message.
_MAX_NODES_LISTED: Final[int] = 12

# ros2_control's own node names. Their presence does not change the verdict —
# a foreign `/joint_states` publisher is refused either way — but it does change
# the message, and "this looks like real hardware" is worth saying out loud.
_HARDWARE_SIGNATURES: Final[tuple[str, ...]] = (
    "controller_manager",
    "joint_state_broadcaster",
    "hardware_interface",
)


def confine_sim_scope(env: dict[str, str]) -> None:
    """Pin a simulation to its own host and its own DDS domain, in place.

    Both settings are skipped when the caller has already exported one, so an
    operator who wants a shared graph keeps it by saying so — the same posture
    ``packages/openral_safety_watchdog/test/conftest.py`` takes for
    ``ROS_DOMAIN_ID``.

    Args:
        env: The launch environment, mutated in place.
    """
    env.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    env.setdefault("ROS_DOMAIN_ID", SIM_DOMAIN_ID)


def _probe_source() -> str:
    """The probe body, run in a subprocess under the launch environment."""
    return f"""
import json, sys, time
try:
    import rclpy
except ImportError:
    json.dump({{"no_ros": True}}, sys.stdout)
    raise SystemExit(0)
rclpy.init(args=None)
node = rclpy.create_node({_PROBE_NODE_NAME!r})
deadline = time.monotonic() + {_PROBE_SPIN_S!r}
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
publishers = [
    {{"node": info.node_name, "namespace": info.node_namespace}}
    for info in node.get_publishers_info_by_topic("/joint_states")
]
names = [
    f"{{ns.rstrip('/')}}/{{name}}"
    for name, ns in node.get_node_names_and_namespaces()
    if name != {_PROBE_NODE_NAME!r}
]
json.dump({{"joint_state_publishers": publishers, "nodes": sorted(names)}}, sys.stdout)
node.destroy_node()
rclpy.shutdown()
"""


def _scan_graph(env: dict[str, str]) -> dict[str, list[object]] | str | None:
    """What is already on the graph.

    Three outcomes, kept distinct because they mean different things:

    * a ``dict`` — the graph was read;
    * :data:`_NO_ROS` — ``rclpy`` is not importable, so there is no graph here
      at all and nothing to collide with;
    * ``None`` — the probe *should* have worked and did not. An unreadable graph
      is **not** treated as an empty one: a probe that failed has not shown the
      graph is clear, and a guard that passes when its instrument is broken is
      worse than no guard.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _probe_source()],
            env=env,
            capture_output=True,
            text=True,
            timeout=_PROBE_SPIN_S + 30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("no_ros"):
        return _NO_ROS
    return parsed


def _looks_like_hardware(nodes: list[str]) -> bool:
    """Whether the occupying graph carries a ros2_control signature."""
    return any(sig in node for node in nodes for sig in _HARDWARE_SIGNATURES)


def assert_graph_unoccupied(env: dict[str, str], *, hal_mode: str) -> None:
    """Refuse to launch onto a ROS graph that already carries a robot.

    Symmetric by construction: the check is the same in both directions, so a
    sim will not start beside a robot and a robot will not start beside a sim.
    ``hal_mode`` only shapes the wording.

    Args:
        env: The launch environment — the scope actually about to be used, not
            the CLI's own.
        hal_mode: ``"sim"`` or ``"real"``.

    Raises:
        ROSConfigError: When a foreign ``/joint_states`` publisher is present,
            or when the graph could not be read at all.
    """
    if os.environ.get(ALLOW_SHARED_GRAPH_ENV) == "1":
        return

    scope = (
        f"ROS_DOMAIN_ID={env.get('ROS_DOMAIN_ID', '0 (unset)')} "
        f"ROS_AUTOMATIC_DISCOVERY_RANGE="
        f"{env.get('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET (unset)')}"
    )
    found = _scan_graph(env)
    if found == _NO_ROS:
        return
    if found is None:
        raise ROSConfigError(
            f"could not read the ROS graph to check it is unoccupied ({scope}). "
            f"Refusing rather than assuming it is clear — a guard that passes "
            f"when its instrument is broken is worse than no guard. Check that "
            f"rclpy imports in this environment, then retry; set "
            f"{ALLOW_SHARED_GRAPH_ENV}=1 to launch without the check.",
        )

    publishers = found.get("joint_state_publishers") or []
    if not publishers:
        return

    nodes = [str(n) for n in (found.get("nodes") or [])]
    owners = ", ".join(
        f"{p.get('namespace', '/')}{p.get('node', '?')}" for p in publishers if isinstance(p, dict)
    )
    other = "real hardware" if _looks_like_hardware(nodes) else "another robot"
    launching = "a simulation" if hal_mode == "sim" else "a real robot"
    raise ROSConfigError(
        f"/joint_states already has {len(publishers)} publisher(s) on this ROS "
        f"graph ({owners}) — you are about to start {launching} on a graph that "
        f"already carries {other}.\n"
        f"  scope: {scope}\n"
        f"  nodes: {', '.join(nodes[:_MAX_NODES_LISTED]) or '(none visible)'}"
        f"{' …' if len(nodes) > _MAX_NODES_LISTED else ''}\n"
        f"A simulation and a robot sharing a graph read each other's "
        f"/joint_states and can reach each other's command topics. On "
        f"2026-09-05 a sim silently consumed a live OpenArm's joint states this "
        f"way for ten rounds (#227).\n"
        f"Run the simulation on a different host or domain, stop the other "
        f"graph, or set {ALLOW_SHARED_GRAPH_ENV}=1 if sharing is intended.",
    )

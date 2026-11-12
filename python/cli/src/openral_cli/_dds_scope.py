# SPDX-License-Identifier: Apache-2.0
"""Keep a simulation and a real robot off each other's ROS graph.

Incident (2026-09-05, issue #227): `openral deploy sim` set no DDS scope, so
it inherited `ROS_DOMAIN_ID` (unset = domain 0, multicast subnet discovery)
and a sim on one host joined a live bimanual OpenArm's graph on another host.
Two `/joint_states` publishers meant the sim's state assembler read OpenArm
joint names instead of `panda_gripper`; a 10-round A/B died in ~50s per round
looking like a policy failure. Nothing actuated only by luck — the robot had
no subscriber on `/openral/candidate_action`; a sim `Action` publish onto a
topic a robot subscribes to is one collision away.

Two controls:

* ``confine_sim_scope`` — prevention: pins a sim to one host + a private
  domain. Measured against a live OpenArm: `ROS_AUTOMATIC_DISCOVERY_RANGE=
  SUBNET` (default) makes its nodes visible, `LOCALHOST` hides them without
  affecting same-host discovery; `OFF` would also hide the sim's own nodes.
* ``confine_real_scope`` — the same prevention for a real robot. Left on
  domain 0 with subnet discovery, anyone on the LAN could publish
  ``/openral/safe_action`` (the HAL obeys it), call ``/openral/estop_reset``
  or spoof ``/joint_states`` and the deadman heartbeat. A real graph that
  must span hosts opts in with ``OPENRAL_ALLOW_NETWORK_ROS=1``.
* ``assert_graph_unoccupied`` — detection: covers a real-robot launch
  joining an existing sim graph, or two runs on one host in the same domain.

Signature: a `/joint_states` publisher — every robot has exactly one.

.. warning::
   The ``ros2`` CLI daemon is unusable here: it answers from its own
   environment, so `ros2 node list` under `LOCALHOST` still reported the
   remote robot's nodes. The probe runs in a subprocess with the exact
   launch environment and talks to ``rclpy`` directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import MutableMapping
from typing import Final

from openral_core.exceptions import ROSConfigError
from openral_core.ros_graph_scope import (
    ALLOW_NETWORK_ROS_ENV,
    discovery_range_is_ros_default,
    require_host_confined_ros_scope,
)

__all__ = [
    "ALLOW_NETWORK_ROS_ENV",
    "ALLOW_UNVERIFIED_GRAPH_ENV",
    "RESERVED_DOMAINS",
    "ROBOT_OPTION_HELP",
    "SIM_DOMAIN_ID",
    "assert_graph_unoccupied",
    "confine_real_scope",
    "confine_sim_scope",
    "hal_node_name",
    "join_robot_scope",
    "join_robot_scope_or_exit",
    "real_domain_id",
    "require_robot_on_graph",
]


#: Escape hatch for **one** of this module's two refusals: the one raised when
#: the graph could not be *read* at all (``rclpy`` will not import, the probe
#: crashed). That is the guard's instrument being broken, not a hazard it
#: detected, and without a way past it a broken rclpy import makes a deploy
#: unrecoverable. Matches the repo's other "I know what I am doing" env gates
#: (``OPENRAL_ALLOW_REMOTE_CODE``, ``OPENRAL_ALLOW_UNSAFE_PICKLE``).
#:
#: It deliberately does **not** cover the occupied-graph refusal. That one is
#: #227 — a sim silently consuming a live OpenArm's joint states for ten rounds
#: — and it is now unconditional: no env var makes OpenRAL start a robot onto a
#: graph that already carries one. The predecessor ``OPENRAL_ALLOW_SHARED_GRAPH``
#: covered both, so the single legitimate need (a broken probe) doubled as a way
#: to switch off a real safety check, and the documented OpenArm bring-up
#: *required* setting it until the vendor ros2_control graph moved inside
#: ``deploy_e2e.launch.py``. A guard whose normal path tells you to disable it
#: teaches operators to disable it (CLAUDE.md §1.1). Setting the old name now
#: does nothing; the occupied-graph refusal names the real remedies instead.
ALLOW_UNVERIFIED_GRAPH_ENV: Final[str] = "OPENRAL_ALLOW_UNVERIFIED_GRAPH"

#: The domain a confined sim runs on. Any fixed value would do — what matters is
#: that it is not 0, which is where an unconfigured robot host lands.
SIM_DOMAIN_ID: Final[str] = "77"

#: Domains the hash table spans. Upper bound 101: on Linux, domains 102-232
#: collide with the default ephemeral port range (ROS 2 "The ROS_DOMAIN_ID"
#: concept page; RTPS port = 7400 + 250 * domain + offsets). Below 10 is skipped
#: (0 = unset, small values are what people pick by hand). The sim's domain was
#: excluded from the start; this table is kept as-is so a robot id that never
#: landed on a reserved value keeps the domain it had.
_HASH_DOMAINS: Final[tuple[int, ...]] = tuple(d for d in range(10, 102) if d != int(SIM_DOMAIN_ID))

#: Domains something else on a robot host already uses, so a real robot is never
#: hashed onto them (a twin, a test or a lab harness on the robot's own domain
#: is on the robot's graph):
#:
#: * ``0`` — the unset default; ``42`` — ``tools/_demo_env.sh``'s default, and
#:   never a robot's by explicit request; ``77`` — ``SIM_DOMAIN_ID``;
#: * ``50-99`` — per-PID test domains (``50 + pid % 50`` in the safety-kernel
#:   launch tests and ``tests/sim/safety``, ``60 + pid % 40`` in the OTel test),
#:   which also cover the lab harnesses' fixed domains (76, 80-89, 85, 87, 88 in
#:   ``docs/reference/dds-large-messages.md``, 90, 91, 92);
#: * ``100-101`` — the bottom of the package test conftests' ``100 + pid % 100``.
#:
#: Exporting ``ROS_DOMAIN_ID`` still wins over all of this.
RESERVED_DOMAINS: Final[frozenset[int]] = frozenset({0, 42, int(SIM_DOMAIN_ID)}) | frozenset(
    range(50, 102)
)

#: Where a robot id whose hash lands on a reserved domain is re-hashed to.
_FREE_DOMAINS: Final[tuple[int, ...]] = tuple(d for d in _HASH_DOMAINS if d not in RESERVED_DOMAINS)

#: How long the probe listens before deciding the graph is empty. DDS discovery
#: is asynchronous: a participant that exists is not necessarily *known* the
#: instant a new one joins. Too short and the guard passes on a graph that is
#: occupied, which is the failure direction that matters.
_PROBE_SPIN_S: Final[float] = 3.0

_PROBE_NODE_NAME: Final[str] = "openral_graph_scope_probe"

#: Returned by ``_scan_graph`` when ``rclpy`` is not importable at all.
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

    An operator's own choice wins, so a shared graph stays possible by saying so
    — the same posture ``packages/openral_safety_watchdog/test/conftest.py``
    takes for ``ROS_DOMAIN_ID``. A non-empty ``ROS_DOMAIN_ID`` is a choice. For
    the range, ``SUBNET`` is not: ROS's ``setup.bash`` exports it into every
    sourced shell, so it is confined like an unset range unless
    ``OPENRAL_ALLOW_NETWORK_ROS=1`` says the LAN is wanted
    (``openral_core.ros_graph_scope.discovery_range_is_ros_default``). An empty
    value is unset to ROS: SUBNET, domain 0.

    Args:
        env: The launch environment, mutated in place.
    """
    # Not setdefault: an exported-but-empty value is unset to ROS (SUBNET, domain 0),
    # and an exported SUBNET is what setup.bash leaves in every sourced shell.
    if discovery_range_is_ros_default(env):
        env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    if not env.get("ROS_DOMAIN_ID", "").strip():
        env["ROS_DOMAIN_ID"] = SIM_DOMAIN_ID


def real_domain_id(robot_id: str) -> str:
    """The ``ROS_DOMAIN_ID`` a real robot runs on when the operator sets none.

    Stable per ``robot_id`` (SHA-256, not ``hash()``, which is salted per
    process), in 10-101 and never in ``RESERVED_DOMAINS``. A hash that lands
    on a reserved domain is re-hashed into the free ones (deterministic, and
    spread rather than piled onto the next free value); every other id keeps
    the domain it had before the reservations, so only ids that used to land on
    a reserved domain moved. Two robots of the same model on one host hash
    alike; they are told apart by the occupancy guard, and an operator
    separates them by exporting ``ROS_DOMAIN_ID``.

    Example:
        >>> real_domain_id("openarm") == real_domain_id("openarm")
        True
        >>> 10 <= int(real_domain_id("openarm")) <= 101
        True
        >>> int(real_domain_id("openarm")) in RESERVED_DOMAINS
        False
    """
    digest = int.from_bytes(hashlib.sha256(robot_id.encode()).digest()[:8], "big")
    domain = _HASH_DOMAINS[digest % len(_HASH_DOMAINS)]
    if domain in RESERVED_DOMAINS:
        domain = _FREE_DOMAINS[digest % len(_FREE_DOMAINS)]
    return str(domain)


def confine_real_scope(env: MutableMapping[str, str], *, robot_id: str) -> None:
    """Keep a real robot's ROS graph on its own host and off domain 0, in place.

    Defaults ``ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`` and
    ``ROS_DOMAIN_ID=real_domain_id(robot_id)``; an exported non-empty domain wins,
    and so does an exported range other than ROS's default (an empty one is what
    ROS reads as unset: SUBNET, domain 0). ``SUBNET`` is that default — ROS's
    ``setup.bash`` exports it into every sourced shell — so it is replaced by
    ``LOCALHOST`` unless ``OPENRAL_ALLOW_NETWORK_ROS=1`` says the LAN is wanted
    (``openral_core.ros_graph_scope.discovery_range_is_ros_default``).
    A scope that reaches other hosts (``ROS_STATIC_PEERS`` set, or a range other
    than ``LOCALHOST`` / ``OFF``) is refused unless
    ``OPENRAL_ALLOW_NETWORK_ROS=1`` is set. An opted-in networked scope is not
    logged here: ``deploy_e2e.launch.py``'s gate prints it once per deploy.

    Args:
        env: The launch environment, mutated in place. The opt-in is read from
            it too, so the decision depends on nothing else.
        robot_id: Seeds the default domain.

    Raises:
        ROSConfigError: A networked scope without the opt-in and SROS2, or a
            secured graph whose keystore does not fit (``require_host_confined_ros_scope``).

    Example:
        >>> env: dict[str, str] = {}
        >>> confine_real_scope(env, robot_id="openarm")
        >>> env["ROS_AUTOMATIC_DISCOVERY_RANGE"]
        'LOCALHOST'
    """
    localhost_only = env.get("ROS_LOCALHOST_ONLY", "").strip() == "1"
    # Static peers do not replace the range: unset is still SUBNET to ROS, so the
    # LOCALHOST default applies with or without them (LOCALHOST + one peer).
    defaulted = not localhost_only and discovery_range_is_ros_default(env)
    # The defaults first, on a copy, so a refusal leaves `env` untouched. Not
    # setdefault: an exported-but-empty value means SUBNET / domain 0 to ROS, and an
    # exported SUBNET is what setup.bash leaves in every sourced shell. The domain is
    # in the copy because a secured graph's keystore is checked against it.
    domain = env.get("ROS_DOMAIN_ID", "").strip() or real_domain_id(robot_id)
    scoped = {**env, "ROS_DOMAIN_ID": domain}
    if defaulted:
        scoped["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    # No warning here: the env reaches deploy_e2e.launch.py, whose gate logs the
    # hazard once per deploy; this runs twice per deploy (orphan reap + launch env, #558).
    require_host_confined_ros_scope(scoped)
    if defaulted:
        env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    env["ROS_DOMAIN_ID"] = domain


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
    * ``_NO_ROS`` — ``rclpy`` is not importable, so there is no graph here
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
    scope = (
        f"ROS_DOMAIN_ID={env.get('ROS_DOMAIN_ID', '0 (unset)')} "
        f"ROS_AUTOMATIC_DISCOVERY_RANGE="
        f"{env.get('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET (unset)')}"
    )
    found = _scan_graph(env)
    if isinstance(found, str):
        # _NO_ROS — see _scan_graph. `isinstance` rather than `== _NO_ROS` so
        # the str arm is narrowed out of the union for the dict access below.
        return
    if found is None:
        # The instrument is broken, not a hazard detected — the one refusal the
        # escape hatch covers. See ALLOW_UNVERIFIED_GRAPH_ENV.
        if os.environ.get(ALLOW_UNVERIFIED_GRAPH_ENV) == "1":
            return
        raise ROSConfigError(
            f"could not read the ROS graph to check it is unoccupied ({scope}). "
            f"Refusing rather than assuming it is clear — a guard that passes "
            f"when its instrument is broken is worse than no guard. Check that "
            f"rclpy imports in this environment, then retry; set "
            f"{ALLOW_UNVERIFIED_GRAPH_ENV}=1 to launch without the check.",
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
        f"There is no env var that waives this. Run the simulation on a "
        f"different host or ROS_DOMAIN_ID, or stop the other graph. If the "
        f"other publisher is the vendor ros2_control graph for the robot you "
        f"are launching, do not start it by hand — deploy_e2e.launch.py starts "
        f"it under hal_mode:=real (the manifest's hal.real_bringup).",
    )


#: ``--robot`` help, shared by ``openral prompt`` and ``openral estop reset``.
ROBOT_OPTION_HELP: Final[str] = (
    "The robot id the deploy runs (`openral deploy run --robot ID`): join its ROS graph — "
    "the per-robot ROS_DOMAIN_ID and LOCALHOST discovery — without exporting them, and "
    "act only if that robot's HAL node (openral_hal_ID) is on it. An exported "
    "ROS_DOMAIN_ID still wins, and is reported when it differs."
)


def hal_node_name(robot_id: str) -> str:
    """The ROS node name a deploy runs ``robot_id``'s HAL under (``deploy_sim._derive_hal_spec``).

    Example:
        >>> hal_node_name("openarm")
        'openral_hal_openarm'
    """
    return f"openral_hal_{robot_id}"


def join_robot_scope(env: MutableMapping[str, str], *, robot_id: str | None) -> str:
    """Put an operator command on the graph of ``openral deploy run --robot ROBOT_ID``, in place.

    ``openral prompt`` and ``openral estop reset`` run ``rclpy`` on their own
    shell's environment. A real deploy runs on ``real_domain_id(robot_id)`` with
    LOCALHOST discovery (``confine_real_scope``), so from a fresh terminal they
    used to land on domain 0 and silently reach nothing (#557). With
    ``robot_id`` this applies ``confine_real_scope`` itself — the deploy's
    function, so the two cannot drift. An exported non-empty ``ROS_DOMAIN_ID``
    still wins, as it does for the deploy: it is the documented way to separate
    two robots of one model, and the deploy started with it is on it too. When
    it differs from the robot's derived domain the returned line says so; the
    commands then act only if the robot's HAL is on that graph
    (``require_robot_on_graph``). Without ``robot_id`` ``env`` is left as it is
    (the dashboard runs ``openral prompt`` inside the deploy's environment).

    Args:
        env: The command's environment (``os.environ``), mutated in place
            before ``rclpy.init()``.
        robot_id: The deploy's ``--robot``; ``None`` keeps the shell's scope.

    Returns:
        One line naming the domain and range used and where the domain came from.

    Raises:
        ROSConfigError: A networked scope without the opt-in
            (``confine_real_scope``), leaving ``env`` untouched — the deploy
            refuses that scope as well.

    Example:
        >>> env = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
        >>> line = join_robot_scope(env, robot_id="openarm")
        >>> env["ROS_DOMAIN_ID"], env["ROS_AUTOMATIC_DISCOVERY_RANGE"]
        ('43', 'LOCALHOST')
        >>> "robot 'openarm'" in line
        True
    """
    exported = env.get("ROS_DOMAIN_ID", "").strip()
    if robot_id:
        confine_real_scope(env, robot_id=robot_id)
        derived = real_domain_id(robot_id)
        if not exported:
            origin = f"robot {robot_id!r}"
        elif exported == derived:
            origin = f"exported, = robot {robot_id!r}"
        else:
            origin = (
                f"exported, overrides robot {robot_id!r}'s {derived}: a deploy started "
                f"without this export is on {derived}"
            )
    else:
        origin = "exported" if exported else "unset; pass --robot <id> to reach a real deploy"
    domain = env.get("ROS_DOMAIN_ID", "").strip() or "0"
    if env.get("ROS_LOCALHOST_ONLY", "").strip() == "1":
        reach = "ROS_LOCALHOST_ONLY=1"
    else:
        reach = "ROS_AUTOMATIC_DISCOVERY_RANGE=" + (
            env.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "").strip() or "SUBNET (unset)"
        )
    return f"ros scope: ROS_DOMAIN_ID={domain} ({origin}) {reach}"


def join_robot_scope_or_exit(command: str, *, robot_id: str | None) -> None:
    """``join_robot_scope`` on ``os.environ`` for a CLI command: print the scope, exit 2 on refusal.

    Call before ``rclpy.init()``: rcl reads the domain and range from the environment.

    Args:
        command: The command's name, prefixed to what it prints.
        robot_id: The command's ``--robot``.

    Raises:
        typer.Exit: Code 2 when ``join_robot_scope`` refuses the scope.
    """
    import typer  # noqa: PLC0415  # reason: only the CLI commands need it

    try:
        typer.echo(f"{command}: {join_robot_scope(os.environ, robot_id=robot_id)}", err=True)
    except ROSConfigError as exc:
        typer.echo(f"{command}: refusing to join robot {robot_id!r}'s ROS graph: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def require_robot_on_graph(
    command: str, node: object, *, robot_id: str | None, timeout_s: float
) -> None:
    """Exit 3 unless ``robot_id``'s HAL node is on the graph ``node`` joined.

    The derived domain names a graph, not a robot: 39 free domains for every
    robot id, so ids collide (``openarm``, ``makermods_metal`` and
    ``openamrobot_v2`` all derive 43), and a typo or an exported domain can land
    on another robot's graph. A deploy runs the HAL as ``openral_hal_<robot_id>``
    (``hal_node_name``), and the occupancy guard keeps one robot per graph, so
    that node being present is the check that the graph is the named robot's.
    Without ``robot_id`` nothing is checked (the shell's scope was chosen by hand).

    Args:
        command: The command's name, prefixed to what it prints.
        node: The command's ``rclpy`` node, already created.
        robot_id: The command's ``--robot``.
        timeout_s: How long to spin for discovery before deciding it is absent.

    Raises:
        typer.Exit: Code 3 when the HAL node never appeared.
    """
    if not robot_id:
        return
    import rclpy  # noqa: PLC0415  # reason: heavy ROS dep deferred
    import typer  # noqa: PLC0415  # reason: only the CLI commands need it

    wanted = hal_node_name(robot_id)
    deadline = time.monotonic() + timeout_s
    names: list[str] = []
    while True:
        names = sorted(name for name, _ns in node.get_node_names_and_namespaces())  # type: ignore[attr-defined]  # reason: rclpy untyped
        if wanted in names or time.monotonic() >= deadline:
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    if wanted in names:
        return
    hals = [n for n in names if n.startswith("openral_hal_")]
    typer.echo(
        f"{command}: {wanted} is not on ROS_DOMAIN_ID="
        f"{os.environ.get('ROS_DOMAIN_ID', '').strip() or '0'} after {timeout_s:.1f} s "
        f"(HAL nodes there: {', '.join(hals) or 'none'}) — this is not robot "
        f"{robot_id!r}'s graph, or its deploy is not up. Nothing sent.",
        err=True,
    )
    raise typer.Exit(code=3)

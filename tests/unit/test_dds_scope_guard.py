# SPDX-License-Identifier: Apache-2.0
"""A simulation and a real robot must not end up on one ROS graph.

On 2026-09-05 they did. `openral deploy sim` set no DDS scope, so it inherited
an unset `ROS_DOMAIN_ID` — domain 0, subnet-wide multicast discovery — and a sim
on one host joined the ROS graph of a **live bimanual OpenArm** on another.
`/joint_states` had two publishers; the sim's state assembler read
`openarm_left_joint1 … openarm_right_joint7` where it wanted `panda_gripper`,
and ten A/B rounds died in ~50 s each looking exactly like policy failures
(#227).

Nothing actuated, and that was luck: the robot's stack used its own topic names
and happened to have no subscriber on `/openral/candidate_action`.

Two controls, tested here, because they cover different directions:

* **confinement** stops a sim reaching out to a robot;
* **the occupancy guard** stops a launch in *either* direction joining a graph a
  robot is already on — which confinement cannot do, since a real robot's graph
  legitimately spans machines and is never confined.

The guard's signature is a foreign ``/joint_states`` publisher, chosen because
every robot has exactly one whether it is real or simulated. One rule, both
directions, and neither side has to recognise the other's node names.

No mocks of OpenRAL code (CLAUDE.md §1.11): these drive the real
:mod:`openral_cli._dds_scope` functions and the real typed exception. The graph
*scan* is the one substituted seam — it is a subprocess boundary talking to a
DDS network, which is exactly what §1.11 permits a double for.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from openral_cli import _dds_scope
from openral_core.exceptions import ROSConfigError


class TestConfineSimScope:
    def test_a_sim_is_pinned_off_the_default_domain_and_to_its_own_host(self) -> None:
        """Unset means domain 0 + SUBNET, which is how the OpenArm was reached."""
        env: dict[str, str] = {}
        _dds_scope.confine_sim_scope(env)
        assert env["ROS_DOMAIN_ID"] == _dds_scope.SIM_DOMAIN_ID
        assert env["ROS_DOMAIN_ID"] != "0"
        assert env["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"

    def test_localhost_not_off_so_the_sims_own_nodes_still_find_each_other(self) -> None:
        """``OFF`` would hide the graph from itself; ``LOCALHOST`` keeps the host.

        Measured against the live robot: ``LOCALHOST`` hid its nodes while
        same-host discovery was unaffected, whereas ``OFF`` sees nothing at all.
        """
        env: dict[str, str] = {}
        _dds_scope.confine_sim_scope(env)
        assert env["ROS_AUTOMATIC_DISCOVERY_RANGE"] != "OFF"

    def test_an_explicit_operator_scope_wins(self) -> None:
        """Attaching a dashboard from another host has to stay possible."""
        env = {"ROS_DOMAIN_ID": "5", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
        _dds_scope.confine_sim_scope(env)
        assert env == {"ROS_DOMAIN_ID": "5", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}


class TestGraphOccupancyGuard:
    """The guard is symmetric: same rule whichever side launches first."""

    _ROBOT_GRAPH: ClassVar[dict[str, list[object]]] = {
        "joint_state_publishers": [{"node": "joint_state_broadcaster", "namespace": "/"}],
        "nodes": [
            "/controller_manager",
            "/joint_state_broadcaster",
            "/openarm_left_hardware_interface",
        ],
    }
    _EMPTY_GRAPH: ClassVar[dict[str, list[object]]] = {"joint_state_publishers": [], "nodes": []}

    def test_a_sim_refuses_to_start_beside_a_robot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 2026-09-05 direction, and it must fail closed rather than read it."""
        monkeypatch.delenv(_dds_scope.ALLOW_SHARED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
        message = str(excinfo.value)
        assert "/joint_states already has 1 publisher" in message
        assert "joint_state_broadcaster" in message, "must name who is already there"
        assert "real hardware" in message, "ros2_control signature must be called out"
        assert _dds_scope.ALLOW_SHARED_GRAPH_ENV in message, "the way out must be named"

    def test_a_real_robot_refuses_to_start_beside_an_existing_graph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reverse direction — the one confinement cannot cover."""
        monkeypatch.delenv(_dds_scope.ALLOW_SHARED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="real")
        assert "a real robot" in str(excinfo.value)

    def test_an_empty_graph_launches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard must not block the normal case."""
        monkeypatch.delenv(_dds_scope.ALLOW_SHARED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._EMPTY_GRAPH)
        _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")

    def test_an_unreadable_graph_refuses_rather_than_assumes_it_is_clear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guard that passes when its instrument is broken is worse than none.

        The same posture the adjudicator takes for an uncertified probe: absence
        of evidence is not evidence of absence.
        """
        monkeypatch.delenv(_dds_scope.ALLOW_SHARED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: None)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
        assert "could not read the ROS graph" in str(excinfo.value)

    def test_the_escape_hatch_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sharing on purpose stays possible, and is opt-IN."""
        monkeypatch.setenv(_dds_scope.ALLOW_SHARED_GRAPH_ENV, "1")
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")

    def test_the_refusal_names_the_scope_it_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the scope the operator cannot tell which graph was scanned."""
        monkeypatch.delenv(_dds_scope.ALLOW_SHARED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        env = {"ROS_DOMAIN_ID": "77", "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied(env, hal_mode="sim")
        assert "ROS_DOMAIN_ID=77" in str(excinfo.value)
        assert "LOCALHOST" in str(excinfo.value)

    def test_an_unset_scope_is_reported_as_the_dangerous_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`domain 0` / `SUBNET` is the combination that caused the incident."""
        monkeypatch.delenv(_dds_scope.ALLOW_SHARED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
        assert "0 (unset)" in str(excinfo.value)
        assert "SUBNET (unset)" in str(excinfo.value)

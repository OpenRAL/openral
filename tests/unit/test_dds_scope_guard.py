# SPDX-License-Identifier: Apache-2.0
"""A simulation and a real robot must not end up on one ROS graph.

On 2026-09-05 they did (#227): `openral deploy sim` set no DDS scope, so it
inherited an unset `ROS_DOMAIN_ID` (domain 0, subnet-wide multicast discovery)
and joined a live bimanual OpenArm's graph on another host. `/joint_states` got
two publishers, the sim's state assembler read `openarm_left_joint1..right_joint7`
where it wanted `panda_gripper`, and ten A/B rounds died in ~50 s each looking
like policy failures. Nothing actuated — luck, not design: the robot's stack
used its own topic names and had no subscriber on `/openral/candidate_action`.

Two controls, covering different directions: **confinement** stops a sim
reaching out to a robot; **the occupancy guard** stops a launch in *either*
direction joining a graph a robot is already on (which confinement can't, since
a real robot's graph legitimately spans machines). The guard's signature is a
foreign `/joint_states` publisher — every robot has exactly one, real or sim.

No mocks of OpenRAL code (CLAUDE.md §1.11): real `openral_cli._dds_scope`
functions and the real typed exception. Only the graph *scan* (a subprocess
boundary to a DDS network) is a substituted seam.
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
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

    def test_an_exported_but_empty_scope_is_treated_as_unset(self) -> None:
        """ROS reads an empty value as the default (SUBNET, domain 0): a compose file
        passing ``VAR=${VAR:-}`` must not leave the sim on the LAN."""
        env = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "", "ROS_DOMAIN_ID": " "}
        _dds_scope.confine_sim_scope(env)
        assert env["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"
        assert env["ROS_DOMAIN_ID"] == _dds_scope.SIM_DOMAIN_ID

    def test_an_explicit_operator_scope_wins(self) -> None:
        """Attaching a dashboard from another host has to stay possible."""
        env = {"ROS_DOMAIN_ID": "5", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
        _dds_scope.confine_sim_scope(env)
        assert env == {"ROS_DOMAIN_ID": "5", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}


class TestConfineRealScope:
    """A real robot's graph stays on its host: every peer can command the arm."""

    def test_an_unset_scope_is_confined_to_this_host_and_off_domain_zero(self) -> None:
        env: dict[str, str] = {}
        _dds_scope.confine_real_scope(env, robot_id="openarm")
        assert env["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"
        assert env["ROS_DOMAIN_ID"] == _dds_scope.real_domain_id("openarm")
        assert env["ROS_DOMAIN_ID"] not in {"0", _dds_scope.SIM_DOMAIN_ID}

    def test_an_exported_but_empty_scope_is_treated_as_unset(self) -> None:
        """ROS reads an empty value as the default — SUBNET, domain 0."""
        env = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "", "ROS_DOMAIN_ID": ""}
        _dds_scope.confine_real_scope(env, robot_id="openarm")
        assert env["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"
        assert env["ROS_DOMAIN_ID"] == _dds_scope.real_domain_id("openarm")

    def test_the_domain_is_stable_and_in_the_safe_range_for_every_robot(self) -> None:
        for robot_id in ("openarm", "so101_follower", "franka_panda", "rizon4", ""):
            domain = _dds_scope.real_domain_id(robot_id)
            assert domain == _dds_scope.real_domain_id(robot_id)
            assert 10 <= int(domain) <= 101
            assert domain != _dds_scope.SIM_DOMAIN_ID

    @settings(max_examples=2000, deadline=None)
    @given(robot_id=st.text())
    def test_no_robot_id_lands_on_a_reserved_domain(self, robot_id: str) -> None:
        """42 by explicit request, plus every domain a sim, test or lab harness uses."""
        domain = int(_dds_scope.real_domain_id(robot_id))
        assert 10 <= domain <= 101
        assert domain not in _dds_scope.RESERVED_DOMAINS
        assert domain != 42

    @settings(max_examples=2000, deadline=None)
    @given(robot_id=st.text())
    def test_ids_that_never_hit_a_reserved_domain_keep_their_domain(self, robot_id: str) -> None:
        """The reservations moved only ids whose old hash was reserved (documented)."""
        digest = int.from_bytes(hashlib.sha256(robot_id.encode()).digest()[:8], "big")
        before = tuple(d for d in range(10, 102) if d != 77)[digest % 91]
        if before not in _dds_scope.RESERVED_DOMAINS:
            assert _dds_scope.real_domain_id(robot_id) == str(before)

    def test_known_robot_domains_are_pinned(self) -> None:
        """openarm kept 43; so101_follower used to hash onto 52 (a test domain), now 39."""
        assert _dds_scope.real_domain_id("openarm") == "43"
        assert _dds_scope.real_domain_id("so101_follower") == "39"

    def test_an_explicit_host_confined_scope_wins(self) -> None:
        env = {"ROS_DOMAIN_ID": "42", "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
        _dds_scope.confine_real_scope(env, robot_id="openarm")
        assert env == {"ROS_DOMAIN_ID": "42", "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}

    @pytest.mark.parametrize(
        "scope",
        [
            {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},
            {"ROS_AUTOMATIC_DISCOVERY_RANGE": "subnet"},
            {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SYSTEM_DEFAULT"},
            {"ROS_STATIC_PEERS": "192.168.1.20"},
            {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_STATIC_PEERS": "10.0.0.2"},
        ],
    )
    def test_a_networked_scope_without_the_opt_in_is_refused(self, scope: dict[str, str]) -> None:
        env = dict(scope)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.confine_real_scope(env, robot_id="openarm")
        assert _dds_scope.ALLOW_NETWORK_ROS_ENV in str(excinfo.value)
        assert "/openral/safe_action" in str(excinfo.value)
        assert env == scope, "a refusal must leave the env untouched"

    def test_the_legacy_localhost_only_flag_is_host_confined(self) -> None:
        """rcl (Jazzy) lets ROS_LOCALHOST_ONLY=1 override the range and the static
        peers ("will be ignored"), so the graph is host-confined: no refusal."""
        env = {
            "ROS_LOCALHOST_ONLY": "1",
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET",
            "ROS_STATIC_PEERS": "10.0.0.2",
        }
        _dds_scope.confine_real_scope(env, robot_id="openarm")
        assert env["ROS_DOMAIN_ID"] == _dds_scope.real_domain_id("openarm")

    def test_localhost_only_zero_does_not_confine(self) -> None:
        env = {"ROS_LOCALHOST_ONLY": "0", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
        with pytest.raises(ROSConfigError):
            _dds_scope.confine_real_scope(env, robot_id="openarm")

    def test_an_opt_in_other_than_one_does_not_count(self) -> None:
        env = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET", _dds_scope.ALLOW_NETWORK_ROS_ENV: "true"}
        with pytest.raises(ROSConfigError):
            _dds_scope.confine_real_scope(env, robot_id="openarm")

    def test_the_opt_in_keeps_the_networked_scope_and_still_leaves_domain_zero(self) -> None:
        env = {
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET",
            _dds_scope.ALLOW_NETWORK_ROS_ENV: "1",
        }
        _dds_scope.confine_real_scope(env, robot_id="openarm")
        assert env["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "SUBNET"
        assert env["ROS_DOMAIN_ID"] == _dds_scope.real_domain_id("openarm")

    def test_a_sim_is_unchanged(self) -> None:
        """The sim keeps its fixed domain; only real robots hash."""
        env: dict[str, str] = {}
        _dds_scope.confine_sim_scope(env)
        assert env == {
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "ROS_DOMAIN_ID": _dds_scope.SIM_DOMAIN_ID,
        }


class TestDeployRunAppliesTheRealScope:
    """``deploy run`` builds its launch env through the confinement, not around it."""

    def test_the_launch_env_is_confined_for_a_real_robot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from openral_cli import deploy_sim

        for name in ("ROS_DOMAIN_ID", "ROS_AUTOMATIC_DISCOVERY_RANGE", "ROS_STATIC_PEERS"):
            monkeypatch.delenv(name, raising=False)
        env = deploy_sim._prepare_launch_env(hal_mode="real", robot_id="openarm")
        assert env["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"
        assert env["ROS_DOMAIN_ID"] == _dds_scope.real_domain_id("openarm")

    def test_the_launch_env_refuses_a_networked_real_robot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from openral_cli import deploy_sim

        monkeypatch.delenv(_dds_scope.ALLOW_NETWORK_ROS_ENV, raising=False)
        monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET")
        with pytest.raises(ROSConfigError):
            deploy_sim._prepare_launch_env(hal_mode="real", robot_id="openarm")


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
        monkeypatch.delenv(_dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
        message = str(excinfo.value)
        assert "/joint_states already has 1 publisher" in message
        assert "joint_state_broadcaster" in message, "must name who is already there"
        assert "real hardware" in message, "ros2_control signature must be called out"
        # The way out must still be named — but it is an action, not an env var.
        assert "different host or ROS_DOMAIN_ID" in message
        assert "stop the other graph" in message

    def test_a_real_robot_refuses_to_start_beside_an_existing_graph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reverse direction — the one confinement cannot cover."""
        monkeypatch.delenv(_dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="real")
        assert "a real robot" in str(excinfo.value)

    def test_an_empty_graph_launches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard must not block the normal case."""
        monkeypatch.delenv(_dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._EMPTY_GRAPH)
        _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")

    def test_an_unreadable_graph_refuses_rather_than_assumes_it_is_clear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guard that passes when its instrument is broken is worse than none.

        The same posture the adjudicator takes for an uncertified probe: absence
        of evidence is not evidence of absence.
        """
        monkeypatch.delenv(_dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: None)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
        assert "could not read the ROS graph" in str(excinfo.value)

    def test_a_host_without_ros_is_not_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ``rclpy`` means no graph to collide with — do not block the launch.

        Distinct from an unreadable graph above, and the distinction is load
        bearing: a host with no ROS cannot run ``ros2 launch`` either, so the
        launch fails on its own with a clearer message than this guard could
        give. Refusing here would block every non-ROS machine (CI included) from
        exercising the launch path while buying no safety, since there is no
        robot within reach.
        """
        monkeypatch.delenv(_dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: _dds_scope._NO_ROS)
        _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")

    def test_no_ros_is_not_confused_with_an_unreadable_graph(self) -> None:
        """The two outcomes must stay distinguishable at the seam itself."""
        assert _dds_scope._NO_ROS is not None
        assert isinstance(_dds_scope._NO_ROS, str)

    def test_the_occupied_graph_refusal_cannot_be_waived(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No env var starts a robot onto a graph that already carries one.

        The predecessor `OPENRAL_ALLOW_SHARED_GRAPH` waived this as well as the
        unreadable-graph refusal, so the one legitimate need (a broken probe)
        doubled as a switch for the #227 guard — and the documented OpenArm
        bring-up *required* setting it, which teaches operators to set it.
        Both spellings are pinned here so neither reopens the hole.
        """
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        for name in ("OPENRAL_ALLOW_SHARED_GRAPH", _dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV):
            monkeypatch.setenv(name, "1")
            with pytest.raises(ROSConfigError):
                _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
            monkeypatch.delenv(name, raising=False)

    def test_the_refusal_offers_no_env_var_as_a_remedy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator must not read "set X=1" as a way past a live robot."""
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
        message = str(excinfo.value)
        assert "OPENRAL_ALLOW_SHARED_GRAPH" not in message
        assert _dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV not in message
        assert "no env var that waives this" in message

    def test_the_escape_hatch_covers_only_an_unreadable_graph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken probe is the instrument failing, not a hazard detected.

        Without a way past it, an environment where `rclpy` will not import
        makes a deploy unrecoverable — so this one refusal stays waivable.
        """
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: None)
        with pytest.raises(ROSConfigError):
            _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
        monkeypatch.setenv(_dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV, "1")
        _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")

    def test_the_refusal_names_the_scope_it_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the scope the operator cannot tell which graph was scanned."""
        monkeypatch.delenv(_dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV, raising=False)
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
        monkeypatch.delenv(_dds_scope.ALLOW_UNVERIFIED_GRAPH_ENV, raising=False)
        monkeypatch.setattr(_dds_scope, "_scan_graph", lambda _env: self._ROBOT_GRAPH)
        with pytest.raises(ROSConfigError) as excinfo:
            _dds_scope.assert_graph_unoccupied({}, hal_mode="sim")
        assert "0 (unset)" in str(excinfo.value)
        assert "SUBNET (unset)" in str(excinfo.value)

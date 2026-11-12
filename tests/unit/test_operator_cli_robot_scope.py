"""`join_robot_scope`: `openral prompt` / `estop reset --robot` join the deploy's graph (#557).

A real deploy runs on ``real_domain_id(robot_id)`` with LOCALHOST discovery. The
operator commands derive the same scope with the deploy's own function and say
which domain they used and where it came from. The live proof that this reaches
a real graph is ``tests/integration/test_operator_cli_robot_domain_live.py``.

No mocks of OpenRAL code (CLAUDE.md §1.11): the real ``openral_cli._dds_scope``
on plain environment dicts, and the real CLI app through Typer's runner.
"""

from __future__ import annotations

import pytest
from openral_cli import _dds_scope
from openral_cli.main import app
from openral_core.exceptions import ROSConfigError
from typer.testing import CliRunner


def test_robot_id_applies_the_deploys_scope_to_a_fresh_shell() -> None:
    env = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}  # setup.bash, nothing exported
    deploy_env = dict(env)
    _dds_scope.confine_real_scope(deploy_env, robot_id="openarm")

    line = _dds_scope.join_robot_scope(env, robot_id="openarm")

    assert env == deploy_env
    assert env["ROS_DOMAIN_ID"] == _dds_scope.real_domain_id("openarm")
    assert line == (
        f"ros scope: ROS_DOMAIN_ID={env['ROS_DOMAIN_ID']} (robot 'openarm') "
        "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST"
    )


def test_an_exported_domain_wins_and_a_mismatch_is_said_out_loud() -> None:
    derived = _dds_scope.real_domain_id("openarm")
    env = {"ROS_DOMAIN_ID": "31"}
    line = _dds_scope.join_robot_scope(env, robot_id="openarm")
    assert env["ROS_DOMAIN_ID"] == "31"
    assert f"(exported, overrides robot 'openarm''s {derived}" in line
    assert f"without this export is on {derived}" in line


def test_an_exported_domain_equal_to_the_robots_is_not_flagged() -> None:
    derived = _dds_scope.real_domain_id("openarm")
    line = _dds_scope.join_robot_scope({"ROS_DOMAIN_ID": derived}, robot_id="openarm")
    assert f"ROS_DOMAIN_ID={derived} (exported, = robot 'openarm')" in line


@pytest.mark.parametrize("robot_id", [None, ""])
def test_without_a_robot_the_shell_scope_is_left_and_named(robot_id: str | None) -> None:
    env = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
    line = _dds_scope.join_robot_scope(env, robot_id=robot_id)
    assert env == {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
    assert line == (
        "ros scope: ROS_DOMAIN_ID=0 (unset; pass --robot <id> to reach a real deploy) "
        "ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET"
    )


def test_localhost_only_is_reported_instead_of_the_range() -> None:
    env = {"ROS_LOCALHOST_ONLY": "1"}
    line = _dds_scope.join_robot_scope(env, robot_id="openarm")
    assert line.endswith(" ROS_LOCALHOST_ONLY=1")


def test_a_networked_scope_is_refused_like_the_deploy_refuses_it() -> None:
    env = {"ROS_STATIC_PEERS": "10.0.0.5"}
    with pytest.raises(ROSConfigError, match=_dds_scope.ALLOW_NETWORK_ROS_ENV):
        _dds_scope.join_robot_scope(env, robot_id="openarm")
    assert env == {"ROS_STATIC_PEERS": "10.0.0.5"}  # a refusal leaves the env alone


@pytest.mark.parametrize("command", [["prompt", "hi"], ["estop", "reset"]])
def test_the_cli_exits_2_before_touching_ros_on_a_refused_scope(
    command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROS_STATIC_PEERS", "10.0.0.5")
    for name in (_dds_scope.ALLOW_NETWORK_ROS_ENV, "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION"):
        monkeypatch.delenv(name, raising=False)
    # Set (to ROS's "unset") so monkeypatch restores them even if the CLI wrote them.
    monkeypatch.setenv("ROS_DOMAIN_ID", "")
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "")
    result = CliRunner().invoke(app, [*command, "--robot", "openarm"])
    assert result.exit_code == 2, result.output
    assert _dds_scope.ALLOW_NETWORK_ROS_ENV in result.output


def test_the_hal_node_name_is_the_one_the_deploy_launches() -> None:
    from openral_cli import deploy_sim

    spec = deploy_sim._derive_hal_spec("openarm", None)
    assert spec.node_name == _dds_scope.hal_node_name("openarm") == "openral_hal_openarm"


class TestDeployHint:
    """The deploy names ``--robot`` only where a fresh shell's ``--robot`` lands on its graph."""

    @staticmethod
    def _printed(env: dict[str, str], hal_mode: str, capsys: pytest.CaptureFixture[str]) -> str:
        from openral_cli import deploy_sim

        deploy_sim._print_ros_scope(env, hal_mode=hal_mode, robot_id="openarm")
        return capsys.readouterr().out

    def test_a_real_robot_on_its_derived_domain_gets_the_hint(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
        _dds_scope.confine_real_scope(env, robot_id="openarm")
        assert "pass them --robot openarm" in self._printed(env, "real", capsys)

    def test_an_exported_domain_or_a_sim_gets_no_hint(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exported = {"ROS_DOMAIN_ID": "31", "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
        assert "--robot" not in self._printed(exported, "real", capsys)
        sim = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
        _dds_scope.confine_sim_scope(sim)
        assert "--robot" not in self._printed(sim, "sim", capsys)

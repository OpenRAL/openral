# SPDX-License-Identifier: Apache-2.0
"""ADR-0130 policy and keystore checks (``openral_core.ros_security``, ``openral security``).

The live DDS-Security behaviour of each enclave is in
``tests/integration/test_sros2_enclaves_live.py``; this covers the policy text
rendered from real robot manifests and the keystore checks, with a real keystore
where one is needed (``sros2_env``, skipped without a sourced ROS 2).
"""

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.ros_security import (
    DEFAULT_ENCLAVE,
    KERNEL_ENCLAVE,
    OPENRAL_ENCLAVES,
    OPERATOR_ENCLAVE,
    default_operator_enclave,
    render_sros2_policy,
    require_secure_ros_graph,
)

from tests.unit.conftest import skip_without_sros2

_REPO = Path(__file__).resolve().parents[2]


def _denied(policy: str, enclave: str, group: str, right: str) -> set[str]:
    """Names under ``<group right="DENY">`` in ``enclave``'s profile."""
    root = ET.fromstring(policy)
    for node in root.iter("enclave"):
        if node.get("path") == enclave:
            return {
                item.text or ""
                for element in node.iter(group)
                if element.get(right) == "DENY"
                for item in element
            }
    raise AssertionError(f"no enclave {enclave}")


def test_every_openral_enclave_is_in_the_policy() -> None:
    root = ET.fromstring(render_sros2_policy({}))
    assert [e.get("path") for e in root.iter("enclave")] == list(OPENRAL_ENCLAVES)


def test_the_manifest_command_and_state_topics_join_the_denials() -> None:
    """Galaxea A1's vendor topics break the ros2_control naming conventions."""
    robot = RobotDescription.from_yaml(str(_REPO / "robots" / "galaxea_a1" / "robot.yaml"))
    policy = render_sros2_policy(robot.onboard_compute)
    denied = _denied(policy, DEFAULT_ENCLAVE, "topics", "publish")
    assert {
        "/arm_joint_target_position",
        "/gripper_position_control_host",
        "/joint_states_host",
    } <= denied
    assert "/arm_joint_target_position" not in _denied(policy, KERNEL_ENCLAVE, "topics", "publish")


def test_only_the_kernel_may_serve_the_estop_reset() -> None:
    """A fake server's success would make `openral estop reset` unlatch the HAL early."""
    policy = render_sros2_policy({})
    for enclave in OPENRAL_ENCLAVES:
        served = "/openral/estop_reset" not in _denied(policy, enclave, "services", "reply")
        assert served == (enclave == KERNEL_ENCLAVE), enclave


def test_the_operator_keeps_the_g1_blend_in_and_nothing_else_on_the_safety_nodes() -> None:
    requests = _denied(render_sros2_policy({}), OPERATOR_ENCLAVE, "services", "request")
    assert "/safety_controller/set_parameters" not in requests
    assert "/safety_controller/change_state" in requests
    assert "/openral_safety_kernel/set_parameters" in requests


def test_a_non_numeric_domain_is_a_typed_refusal(tmp_path: Path) -> None:
    env = {
        "ROS_SECURITY_ENABLE": "true",
        "ROS_SECURITY_STRATEGY": "Enforce",
        "ROS_SECURITY_KEYSTORE": str(tmp_path),
        "ROS_DOMAIN_ID": "4 3",
    }
    with pytest.raises(ROSConfigError):
        require_secure_ros_graph(env)


def test_a_non_numeric_domain_is_refused_with_a_real_keystore(
    sros2_env: Callable[[str], dict[str, str]],
) -> None:
    with pytest.raises(ROSConfigError, match="not a domain number"):
        require_secure_ros_graph({**sros2_env("23"), "ROS_DOMAIN_ID": "abc"})


def test_an_exported_enclave_wins_over_the_operator_default() -> None:
    env = {"ROS_SECURITY_ENABLE": "true", "ROS_SECURITY_ENCLAVE_OVERRIDE": "/openral/hal"}
    default_operator_enclave(env)
    assert env["ROS_SECURITY_ENCLAVE_OVERRIDE"] == "/openral/hal"
    unsecured: dict[str, str] = {}
    default_operator_enclave(unsecured)
    assert unsecured == {}


def test_force_regenerates_and_a_refused_init_keeps_the_keystore(tmp_path: Path) -> None:
    """`--force` swaps the new keystore in only once it is complete."""
    from openral_cli.main import app
    from typer.testing import CliRunner

    skip_without_sros2()
    keystore = tmp_path / "openarm"
    runner = CliRunner()
    args = ["security", "init", "openarm", "--keystore", str(keystore), "--domain-id", "43"]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    old_cert = (keystore / "enclaves" / "cert.pem").read_bytes()

    refused = runner.invoke(app, args)
    assert refused.exit_code == 1
    assert (keystore / "enclaves" / "cert.pem").read_bytes() == old_cert

    forced = runner.invoke(app, [*args, "--force"])
    assert forced.exit_code == 0, forced.output
    assert (keystore / "enclaves" / "cert.pem").read_bytes() != old_cert
    assert not (tmp_path / ".openarm.new").exists()
    require_secure_ros_graph(
        {
            "ROS_SECURITY_ENABLE": "true",
            "ROS_SECURITY_STRATEGY": "Enforce",
            "ROS_SECURITY_KEYSTORE": str(keystore),
            "ROS_DOMAIN_ID": "43",
        }
    )


def _secured(keystore: Path, domain: str) -> dict[str, str]:
    return {
        "ROS_SECURITY_ENABLE": "true",
        "ROS_SECURITY_STRATEGY": "Enforce",
        "ROS_SECURITY_KEYSTORE": str(keystore),
        "ROS_DOMAIN_ID": domain,
    }


def test_the_domain_is_read_from_the_signed_governance(
    sros2_env: Callable[[str], dict[str, str]],
) -> None:
    """DDS loads governance.p7s; an edited governance.xml must not change the answer."""
    env = sros2_env("31")
    keystore = Path(env["ROS_SECURITY_KEYSTORE"])
    unsigned = keystore / "enclaves" / "governance.xml"
    original = unsigned.read_text()
    unsigned.write_text(original.replace("<id>31</id>", "<id>32</id>"))
    try:
        with pytest.raises(ROSConfigError, match="not ROS_DOMAIN_ID=32"):
            require_secure_ros_graph(_secured(keystore, "32"))
        require_secure_ros_graph(_secured(keystore, "31"))
    finally:
        unsigned.write_text(original)


def test_an_allow_everything_keystore_is_refused(tmp_path: Path) -> None:
    """`ros2 security create_enclave` signs allow-all permissions: valid, but not a policy."""
    skip_without_sros2()
    keystore = tmp_path / "plain"
    env = {**os.environ, "ROS_DOMAIN_ID": "33"}
    subprocess.run(
        ["ros2", "security", "create_keystore", str(keystore)],
        env=env,
        capture_output=True,
        check=True,
    )
    for enclave in OPENRAL_ENCLAVES:
        subprocess.run(
            ["ros2", "security", "create_enclave", str(keystore), enclave],
            env=env,
            capture_output=True,
            check=True,
        )
    with pytest.raises(ROSConfigError, match="not an OpenRAL policy"):
        require_secure_ros_graph(_secured(keystore, "33"))

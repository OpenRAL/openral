# SPDX-License-Identifier: Apache-2.0
"""SROS2 (DDS-Security) enclaves for an OpenRAL graph (ADR-0130).

A networked real cell runs under DDS-Security in Enforce mode: every
participant authenticates with an enclave's certificate and is held to that
enclave's signed permissions. The policy (rendered per robot by
``openral security init``) denies the safety surfaces — publishing
``/openral/safe_action`` or a controller's command topic, calling
``/openral/estop_reset``, cycling or re-parameterising the kernel, HAL and
E-stop producers — to every enclave that does not need them.

This module holds the enclave names and the keystore check shared by the
scope gate (``ros_graph_scope``), ``openral deploy`` and the deploy launch.
See ``docs/reference/ros-graph-security.md``.
"""

from __future__ import annotations

import email
import email.message
import xml.etree.ElementTree as ET
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Final

from openral_core.exceptions import ROSConfigError

__all__ = [
    "DEFAULT_ENCLAVE",
    "ENCLAVE_OVERRIDE_ENV",
    "HAL_ENCLAVE",
    "KERNEL_ENCLAVE",
    "LAUNCH_ENCLAVE",
    "OPENRAL_ENCLAVES",
    "OPERATOR_ENCLAVE",
    "default_operator_enclave",
    "render_sros2_policy",
    "require_secure_ros_graph",
    "ros_security_enabled",
]

#: rcl reads it before ``--enclave``, so it pins one process's enclave.
ENCLAVE_OVERRIDE_ENV: Final[str] = "ROS_SECURITY_ENCLAVE_OVERRIDE"

#: Every process the launch does not name: vendor drivers, perception, peers.
DEFAULT_ENCLAVE: Final[str] = "/"
#: The C++ safety kernel, the only publisher of ``/openral/safe_action``.
KERNEL_ENCLAVE: Final[str] = "/openral/safety_kernel"
#: The HAL plus the vendor ``ros2_control`` bringup and scene drivers.
HAL_ENCLAVE: Final[str] = "/openral/hal"
#: ``ros2 launch`` itself and its lifecycle autostart subprocesses.
LAUNCH_ENCLAVE: Final[str] = "/openral/launch"
#: The dashboard, ``openral estop reset`` and ``openral prompt``.
OPERATOR_ENCLAVE: Final[str] = "/openral/operator"

OPENRAL_ENCLAVES: Final[tuple[str, ...]] = (
    DEFAULT_ENCLAVE,
    KERNEL_ENCLAVE,
    HAL_ENCLAVE,
    LAUNCH_ENCLAVE,
    OPERATOR_ENCLAVE,
)

#: A keystore enclave directory is usable only with all of these.
_ENCLAVE_FILES: Final[tuple[str, ...]] = (
    "cert.pem",
    "key.pem",
    "governance.p7s",
    "identity_ca.cert.pem",
    "permissions_ca.cert.pem",
    "permissions.p7s",
)


def ros_security_enabled(env: Mapping[str, str]) -> bool:
    """Whether rcl runs ``env``'s nodes under DDS-Security (exactly ``"true"``, as rcl reads it).

    Example:
        >>> ros_security_enabled({"ROS_SECURITY_ENABLE": "true"})
        True
        >>> ros_security_enabled({"ROS_SECURITY_ENABLE": "1"})
        False
    """
    return env.get("ROS_SECURITY_ENABLE") == "true"


#: Writes every enclave but the kernel's must be denied in its *signed* permissions, so a
#: hand-made allow-everything keystore (``ros2 security create_enclave``) is refused.
_REQUIRED_DENIALS: Final[dict[str, frozenset[str]]] = {
    enclave: frozenset(
        {"rt/openral/safe_action"}
        | (
            {"rq/openral/estop_resetRequest", "rq/openral_safety_kernel/change_stateRequest"}
            if enclave == DEFAULT_ENCLAVE
            else set()
        )
    )
    for enclave in OPENRAL_ENCLAVES
    if enclave != KERNEL_ENCLAVE
}


def _signed_xml(p7s: Path) -> ET.Element:
    """The XML a signed keystore document carries: what DDS loads, not the unsigned copy.

    sros2 and ``openssl smime -sign -text`` write S/MIME ``multipart/signed``; the
    first part is the document. The signature itself is checked by DDS against the
    keystore's CA at participant creation.
    """
    message = email.message_from_bytes(p7s.read_bytes())
    if not message.is_multipart():
        raise ValueError("not an S/MIME multipart/signed document")
    part = message.get_payload(0)
    if not isinstance(part, email.message.Message):
        raise ValueError("signed document has no first part")
    body = part.get_payload(decode=True)
    if not isinstance(body, bytes):
        raise ValueError("signed part has no body")
    return ET.fromstring(body)


def _governance_domains(governance: ET.Element) -> set[int]:
    """The domain ids a governance document admits (``<id>`` and ``<id_range>``)."""
    domains: set[int] = set()
    for rule in governance.iter("domains"):
        for child in rule:
            if child.tag == "id" and child.text:
                domains.add(int(child.text))
            elif child.tag == "id_range":
                low = int(child.findtext("min", "0"))
                domains.update(range(low, int(child.findtext("max", str(low))) + 1))
    return domains


def require_secure_ros_graph(env: Mapping[str, str]) -> Path:
    """Refuse unless ``env`` runs the graph under SROS2 Enforce with a usable OpenRAL keystore.

    Usable: ``ROS_SECURITY_KEYSTORE`` holds every enclave in ``OPENRAL_ENCLAVES``
    with its keys and signed files, its signed governance admits ``ROS_DOMAIN_ID``
    (unset = 0), and every enclave but the kernel's has signed permissions that
    deny publishing ``/openral/safe_action`` (the default enclave also the E-stop
    reset and the kernel's ``change_state``). rcl would otherwise fail one node at
    a time, a node in a missing enclave would silently not join, or an
    allow-everything keystore would pass for a policy.

    Returns:
        The keystore path.

    Raises:
        ROSConfigError: Security off, Permissive, or a missing/incomplete keystore.

    Example:
        >>> try:
        ...     require_secure_ros_graph({"ROS_SECURITY_ENABLE": "false"})
        ... except ROSConfigError:
        ...     print("refused")
        refused
    """
    remedy = "Run `openral security init <robot_id>` and export what it prints."
    if not ros_security_enabled(env) or env.get("ROS_SECURITY_STRATEGY") != "Enforce":
        raise ROSConfigError(
            "SROS2 is required here: export ROS_SECURITY_ENABLE=true and "
            "ROS_SECURITY_STRATEGY=Enforce (got "
            f"ROS_SECURITY_ENABLE={env.get('ROS_SECURITY_ENABLE', '(unset)')}, "
            f"ROS_SECURITY_STRATEGY={env.get('ROS_SECURITY_STRATEGY', '(unset)')}). {remedy}"
        )
    raw = env.get("ROS_SECURITY_KEYSTORE", "").strip()
    if not raw:
        raise ROSConfigError(f"ROS_SECURITY_KEYSTORE is not set. {remedy}")
    keystore = Path(raw).expanduser()
    enclaves = keystore / "enclaves"
    for enclave in OPENRAL_ENCLAVES:
        directory = enclaves / enclave.lstrip("/")
        missing = [f for f in _ENCLAVE_FILES if not (directory / f).is_file()]
        if missing:
            raise ROSConfigError(
                f"keystore {keystore} has no usable enclave {enclave!r} "
                f"(missing {', '.join(missing)} in {directory}). {remedy}"
            )
    governance = enclaves / "governance.p7s"
    try:
        domains = _governance_domains(_signed_xml(governance))
        for enclave, required in _REQUIRED_DENIALS.items():
            signed = enclaves / enclave.lstrip("/") / "permissions.p7s"
            denied = {
                t.text for t in _signed_xml(signed).iterfind(".//deny_rule/publish/topics/topic")
            }
            if missing_denials := sorted(required - denied):
                raise ROSConfigError(
                    f"keystore {keystore}: enclave {enclave!r}'s signed permissions do not "
                    f"deny {missing_denials}; it is not an OpenRAL policy. {remedy}"
                )
    except (OSError, ET.ParseError, ValueError) as exc:
        raise ROSConfigError(f"cannot read keystore {keystore}: {exc}. {remedy}") from exc
    raw_domain = env.get("ROS_DOMAIN_ID", "").strip() or "0"
    if not raw_domain.isdigit():
        raise ROSConfigError(f"ROS_DOMAIN_ID={raw_domain!r} is not a domain number.")
    domain = int(raw_domain)
    if domain not in domains:
        raise ROSConfigError(
            f"keystore {keystore} admits ROS domain(s) {sorted(domains)}, not "
            f"ROS_DOMAIN_ID={domain}; every node would fail to join. Regenerate it for this "
            f"domain: `ROS_DOMAIN_ID={domain} openral security init <robot_id> --force`."
        )
    return keystore


def default_operator_enclave(env: MutableMapping[str, str]) -> None:
    """Put an operator CLI (estop reset, prompt, dashboard) in ``OPERATOR_ENCLAVE`` when secured.

    An exported override wins. Without security the variable is left alone.

    Example:
        >>> env = {"ROS_SECURITY_ENABLE": "true"}
        >>> default_operator_enclave(env)
        >>> env[ENCLAVE_OVERRIDE_ENV]
        '/openral/operator'
    """
    if ros_security_enabled(env):
        env.setdefault(ENCLAVE_OVERRIDE_ENV, OPERATOR_ENCLAVE)


#: Publishing these moves or releases the robot without the kernel: the kernel's
#: output, the stop release, and the ``ros2_control`` command conventions.
_COMMAND_TOPICS: Final[tuple[str, ...]] = (
    "/openral/safe_action",
    "/openral/estop_cleared",
    "/*/commands",
    "/*/joint_trajectory",
)
_COMMAND_ACTIONS: Final[tuple[str, ...]] = ("/*/follow_joint_trajectory",)
#: The joint states the kernel and the HAL check commands against.
_STATE_TOPICS: Final[tuple[str, ...]] = ("/joint_states", "/openral_hal_*/joint_states")
#: Nodes whose lifecycle or parameters gate a stop, a limit or a motor.
_PROTECTED_NODES: Final[tuple[str, ...]] = (
    "openral_safety_kernel",
    "openral_hal_*",
    "openral_deadman_watchdog",
    "openral_hardware_estop",
    "openral_human_estop_forwarder",
    "openral_compliance_recorder",
    "safety_controller",
)
_NODE_SERVICES: Final[tuple[str, ...]] = (
    "change_state",
    "set_parameters",
    "set_parameters_atomically",
)
_OPERATOR_PARAMETER_SERVICES: Final[frozenset[str]] = frozenset(
    {"/safety_controller/set_parameters", "/safety_controller/set_parameters_atomically"}
)
#: ``onboard_compute`` keys that name a vendor command or state topic.
_MANIFEST_COMMAND_KEYS: Final[tuple[str, ...]] = ("command_topic", "gripper_command_topic")
_MANIFEST_STATE_KEYS: Final[tuple[str, ...]] = ("joint_state_topic",)


def _manifest_topics(onboard_compute: Mapping[str, object], keys: tuple[str, ...]) -> list[str]:
    return [
        value
        for key in keys
        if isinstance(value := onboard_compute.get(key), str) and value.startswith("/")
    ]


def render_sros2_policy(onboard_compute: Mapping[str, object]) -> str:
    """Render the sros2 policy (0.2.0) for one robot: the ADR-0130 enclave table.

    Each enclave denies the protected surfaces it does not need, as a client and,
    for ``estop_reset``, the controller manager and the trajectory action, as a
    server too; then allows the rest. sros2 emits deny rules before allow rules,
    so the denials win.

    Args:
        onboard_compute: The robot's ``RobotDescription.onboard_compute``; its
            ``command_topic`` / ``gripper_command_topic`` join the command topics
            and ``joint_state_topic`` the state topics.

    Returns:
        The policy XML, for ``ros2 security generate_artifacts -p``.

    Example:
        >>> xml = render_sros2_policy({"command_topic": "/arm/joint_target"})
        >>> "<topic>/arm/joint_target</topic>" in xml
        True
    """
    commands = [*_COMMAND_TOPICS, *_manifest_topics(onboard_compute, _MANIFEST_COMMAND_KEYS)]
    states = [*_STATE_TOPICS, *_manifest_topics(onboard_compute, _MANIFEST_STATE_KEYS)]
    node_services = [f"/{node}/{srv}" for node in _PROTECTED_NODES for srv in _NODE_SERVICES]
    # The G1 operator blends the S0 policy in with `ros2 param set /safety_controller
    # blend_ratio` (openral_hal_unitree README); only the operator keeps that.
    operator_node_services = [s for s in node_services if s not in _OPERATOR_PARAMETER_SERVICES]
    controller = ["/controller_manager/*"]
    reset = ["/openral/estop_reset"]
    hal_only = [t for t in commands if t not in ("/openral/safe_action", "/openral/estop_cleared")]
    actions = list(_COMMAND_ACTIONS)
    # Per enclave: publish-denied topics, request-denied services, reply-denied services
    # (serving them would let the enclave answer the real server's callers first: a fake
    # estop_reset "success" makes `openral estop reset` broadcast estop_cleared while the
    # kernel stays latched), call-denied and execute-denied actions. Only the kernel
    # serves estop_reset; only the HAL's vendor stack serves the controller manager and
    # the trajectory action.
    denied: dict[str, tuple[list[str], ...]] = {
        KERNEL_ENCLAVE: ([], [], [], [], []),
        HAL_ENCLAVE: (
            ["/openral/safe_action", "/openral/estop_cleared"],
            reset + node_services,
            reset,
            [],
            [],
        ),
        LAUNCH_ENCLAVE: (commands + states, [], reset + controller, actions, actions),
        OPERATOR_ENCLAVE: (
            ["/openral/safe_action", *hal_only, *states],
            operator_node_services + controller,
            reset + controller,
            actions,
            actions,
        ),
        DEFAULT_ENCLAVE: (
            commands + states,
            reset + node_services + controller,
            reset + controller,
            actions,
            actions,
        ),
    }
    root = ET.Element("policy", version="0.2.0")
    enclaves = ET.SubElement(root, "enclaves")
    for path in OPENRAL_ENCLAVES:
        topics, requests, replies, calls, executes = denied[path]
        profile = ET.SubElement(
            ET.SubElement(ET.SubElement(enclaves, "enclave", path=path), "profiles"),
            "profile",
            ns="/",
            node="openral",
        )
        if topics:
            _names(profile, "topics", "topic", topics, publish="DENY")
        if requests:
            _names(profile, "services", "service", requests, request="DENY")
        if replies:
            _names(profile, "services", "service", replies, reply="DENY")
        if calls:
            _names(profile, "actions", "action", calls, call="DENY")
        if executes:
            _names(profile, "actions", "action", executes, execute="DENY")
        _names(profile, "topics", "topic", ["/*"], publish="ALLOW", subscribe="ALLOW")
        _names(profile, "services", "service", ["/*"], request="ALLOW", reply="ALLOW")
        _names(profile, "actions", "action", ["/*"], call="ALLOW", execute="ALLOW")
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def _names(parent: ET.Element, group: str, item: str, names: list[str], **rights: str) -> None:
    element = ET.SubElement(parent, group, rights)
    for name in names:
        ET.SubElement(element, item).text = name

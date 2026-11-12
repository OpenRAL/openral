"""``openral security init`` — one SROS2 keystore per robot (ADR-0130).

A networked real cell runs under DDS-Security in Enforce mode. This command
renders the robot's policy (``openral_core.ros_security.render_sros2_policy``:
the five OpenRAL enclaves, each denied the safety surfaces it does not need,
plus the manifest's vendor command topics) and has ``ros2 security
generate_artifacts`` turn it into keys and signed permissions for the domain
the deploy will run on.

Requires a sourced ROS 2 install with ``sros2``.
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import typer

__all__ = ["default_keystore", "generate_keystore", "security_app"]

security_app = typer.Typer(
    name="security",
    help="SROS2 keystores for a networked real cell. Requires a sourced ROS 2 install.",
    no_args_is_help=True,
)


def default_keystore(robot_id: str) -> Path:
    """``$XDG_DATA_HOME/openral/sros2/<robot_id>`` (``~/.local/share`` when unset).

    Example:
        >>> default_keystore("openarm").parts[-3:]
        ('openral', 'sros2', 'openarm')
    """
    data = os.environ.get("XDG_DATA_HOME", "").strip() or str(Path.home() / ".local" / "share")
    return Path(data) / "openral" / "sros2" / robot_id


def generate_keystore(policy_xml: str, keystore: Path, *, domain: str) -> None:
    """Write a keystore at ``keystore`` (must not exist) for ``policy_xml`` on ROS ``domain``.

    Runs the system ``ros2 security generate_artifacts``, not the venv
    interpreter: sros2's CLI needs ROS's own Python deps (argcomplete, lxml).
    Each enclave's permissions then get ``_allow_the_other_side`` and are
    re-signed with the keystore's permissions CA (``openssl smime``, the format
    sros2 writes). The policy is kept as ``policy.xml`` beside the keys for
    audit; private keys are made owner-only.

    Raises:
        ROSConfigError: ``ros2`` is not on PATH, or sros2 refused; nothing is left behind.
    """
    from openral_core.exceptions import ROSConfigError

    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise ROSConfigError("`ros2` is not on PATH; source ROS 2 (with sros2) first.")
    keystore.mkdir(parents=True, mode=0o700)
    policy = keystore / "policy.xml"
    policy.write_text(policy_xml)
    env = {
        **os.environ,
        "ROS_DOMAIN_ID": domain,
        # sros2 adds the `ros_discovery_info` rule only for an RMW that needs it;
        # Cyclone and Fast DDS both do, so either default is fine.
        "RMW_IMPLEMENTATION": os.environ.get("RMW_IMPLEMENTATION") or "rmw_cyclonedds_cpp",
    }
    cmd = [ros2, "security", "generate_artifacts", "-k", str(keystore), "-p", str(policy)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    try:
        if result.returncode != 0:
            raise ROSConfigError(
                f"`{' '.join(cmd)}` failed ({result.returncode}):\n{result.stdout}{result.stderr}"
            )
        for permissions in (keystore / "enclaves").rglob("permissions.xml"):
            _allow_the_other_side(permissions)
            _sign(permissions, keystore)
    except BaseException:
        shutil.rmtree(keystore)  # a half-made keystore would pass for a usable one
        raise
    # The private keys sign every enclave's identity; only the deploy user reads them.
    for key in keystore.rglob("*key.pem"):
        key.chmod(0o600)


def _allow_the_other_side(permissions: Path) -> None:
    """Grant the side of each denied surface that stays allowed, ahead of the deny rule.

    Cyclone DDS (0.10, Jazzy's default RMW) refuses to create a *topic* that any
    deny rule names, even a publish-only one: denying ``/openral/safe_action``
    publication stopped the HAL subscribing to it, and denying the watchdog's
    ``set_parameters`` requests stopped the watchdog serving them. Fast DDS
    follows the DDS-Security spec and does not. An allow rule listed first for
    the opposite direction (subscribe what is denied for publishing, publish
    what is denied for subscribing — a service's request reader and reply
    writer) lets Cyclone create the topic; the deny rule still refuses the
    denied writer or reader on both RMWs. Measured 2026-11-12, both RMWs.
    """
    tree = ET.parse(permissions)
    for grant in tree.getroot().iter("grant"):
        deny = grant.find("deny_rule")
        if deny is None:
            continue
        allow = ET.Element("allow_rule")
        domains = deny.find("domains")
        if domains is not None:
            allow.append(copy.deepcopy(domains))
        for denied, granted in (("subscribe", "publish"), ("publish", "subscribe")):
            topics = [t.text for t in deny.iterfind(f"{denied}/topics/topic") if t.text]
            if topics:
                group = ET.SubElement(ET.SubElement(allow, granted), "topics")
                for topic in topics:
                    ET.SubElement(group, "topic").text = topic
        grant.insert(list(grant).index(deny), allow)
    tree.write(permissions, encoding="unicode")


def _sign(permissions: Path, keystore: Path) -> None:
    """Re-sign ``permissions.xml`` as ``permissions.p7s`` with the keystore's permissions CA."""
    from openral_core.exceptions import ROSConfigError

    openssl = shutil.which("openssl")
    if openssl is None:
        raise ROSConfigError("`openssl` is not on PATH; it signs the enclave permissions.")
    cmd = [
        openssl,
        "smime",
        "-sign",
        "-text",
        "-in",
        str(permissions),
        "-out",
        str(permissions.with_suffix(".p7s")),
        "-signer",
        str(keystore / "public" / "permissions_ca.cert.pem"),
        "-inkey",
        str(keystore / "private" / "permissions_ca.key.pem"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ROSConfigError(f"signing {permissions} failed: {result.stderr}")


@security_app.command("init")
def security_init_command(
    robot_id: str = typer.Argument(..., help="Robot id (the directory under robots/)."),
    keystore: Path | None = typer.Option(
        None, "--keystore", help="Keystore directory (default: $XDG_DATA_HOME/openral/sros2/<id>)."
    ),
    domain_id: int | None = typer.Option(
        None,
        "--domain-id",
        help=(
            "ROS domain the keystore admits. Default: the exported ROS_DOMAIN_ID, else the "
            "robot's `openral deploy run` domain."
        ),
    ),
    force: bool = typer.Option(
        False, "--force", help="Delete and regenerate an existing keystore."
    ),
) -> None:
    """Generate the robot's SROS2 keystore: keys, governance and per-enclave permissions."""
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSConfigError
    from openral_core.ros_security import (
        render_sros2_policy,
        require_secure_ros_graph,
    )
    from openral_sim.policies.robots import (
        resolve_robot_manifest,
    )

    from openral_cli._dds_scope import real_domain_id
    from openral_cli.deploy_sim import _repo_root_from

    description = RobotDescription.from_yaml(
        str(resolve_robot_manifest(robot_id, repo_root=_repo_root_from(Path(__file__))))
    )
    exported = os.environ.get("ROS_DOMAIN_ID", "").strip()
    domain = str(domain_id) if domain_id is not None else exported or real_domain_id(robot_id)
    path = (keystore or default_keystore(robot_id)).expanduser().resolve()
    if path.exists():
        if not force:
            typer.echo(
                f"openral security init: {path} exists; pass --force to regenerate.", err=True
            )
            raise typer.Exit(code=1)
        shutil.rmtree(path)
    try:
        generate_keystore(render_sros2_policy(description.onboard_compute), path, domain=domain)
    except ROSConfigError as exc:
        typer.echo(f"openral security init: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    exports = {
        "ROS_DOMAIN_ID": domain,
        "ROS_SECURITY_KEYSTORE": str(path),
        "ROS_SECURITY_ENABLE": "true",
        "ROS_SECURITY_STRATEGY": "Enforce",
    }
    require_secure_ros_graph(exports)
    typer.echo(f"openral security init: keystore for {robot_id} (domain {domain}) at {path}")
    typer.echo("export " + " ".join(f"{k}={v}" for k, v in exports.items()))

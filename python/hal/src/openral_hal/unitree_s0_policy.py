"""The Unitree hg S0 policy files — one typed reader for the HAL and the bringup launch.

A real Unitree hg humanoid's legs belong to a whole-body ONNX policy that NVIDIA's
``InferenceController`` runs inside ``ros2_control`` (see ``openral_hal.unitree_real``).
*Which* policy, and therefore which joints it owns, is chosen per deployment from
``packages/openral_hal_unitree/config/<model>/policies/<policy>.yaml``:

* ``openral_policy`` — metadata: ``status``, where the LEAPP bundle comes from
  (``data_package`` share, or a locally built ``bundle_dir``), the velocity-command topic
  and wire type, and ``s0_joints`` — the joints the policy commands;
* ``controllers`` — ROS parameter blocks the launch loads after the policy-independent
  ``controller_manager.yaml``; ``upper_body_forward_joint_command_controller.joints`` is
  what OpenRAL commands.

Both ``UnitreeHgRealHAL`` (which joints to refuse) and ``real_bringup.launch.py`` (which
controllers to configure) read the same file through this module, so the joint split the
HAL enforces is by construction the one the C++ graph runs.

Example:
    >>> from openral_hal.unitree_s0_policy import load_s0_policy, s0_policy_path
    >>> policy = load_s0_policy(s0_policy_path("g1", "gr00t_decoupled_wbc"))
    >>> len(policy.openral_policy.s0_joints), len(policy.upper_body_joints)
    (15, 14)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from openral_core.exceptions import ROSConfigError
from pydantic import BaseModel, ConfigDict, ValidationError

__all__ = [
    "S0PolicyFile",
    "S0PolicyMeta",
    "load_s0_policy",
    "policy_parameters",
    "require_watchdog_enforced",
    "resolve_leapp_config",
    "s0_policy_path",
]

#: The ament package the policy files are installed with.
UNITREE_BRINGUP_PACKAGE = "openral_hal_unitree"
#: The ForwardJointCommandController OpenRAL commands; its ``joints`` are the upper body.
UPPER_BODY_CONTROLLER = "upper_body_forward_joint_command_controller"
#: The S0 policy controller.
INFERENCE_CONTROLLER = "inference_controller"
#: The only inference controller type that enforces ``WATCHDOG_KEYS`` (issue #565).
GUARDED_INFERENCE_CONTROLLER = "openral_hal_unitree/GuardedInferenceController"
#: Stale-input watchdog parameters: isaac_ros_deploy v5.0 names, absent from v4.6.
WATCHDOG_KEYS = ("topic_input_timeout_ms", "topic_input_sources")
#: Repo-checkout location of the configs (``python/hal/src/openral_hal/`` → repo root).
_REPO_CONFIG = Path(__file__).resolve().parents[4] / "packages" / UNITREE_BRINGUP_PACKAGE / "config"


class S0PolicyMeta(BaseModel):
    """``openral_policy`` block of one S0 policy file.

    Attributes:
        name: Policy id (the file stem).
        status: ``ready`` or ``placeholder``; only ``ready`` policies are started.
        data_package: ament package whose ``share/<pkg>/data/`` holds the LEAPP bundle, or
            ``None`` when the bundle is built locally into ``bundle_dir``.
        bundle_dir: Directory (``~`` expanded) holding a locally built bundle.
        leapp_config: The bundle's LEAPP YAML file name.
        velocity_command_topic: Where the policy reads ``command/body/velocity``.
        velocity_command_type: ROS type of that topic.
        s0_joints: Joints the policy's action commands — never commanded by OpenRAL.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: Literal["ready", "placeholder"]
    data_package: str | None = None
    bundle_dir: str | None = None
    leapp_config: str
    velocity_command_topic: str
    velocity_command_type: str
    s0_joints: list[str]


class S0PolicyFile(BaseModel):
    """One ``config/<model>/policies/<policy>.yaml``.

    Raises:
        pydantic.ValidationError: Malformed file (wrapped as ``ROSConfigError`` by
            ``load_s0_policy``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    openral_policy: S0PolicyMeta
    controllers: dict[str, Any]

    @property
    def upper_body_joints(self) -> list[str]:
        """Joints OpenRAL commands (the upper-body controller's ``joints``)."""
        params = self.controllers.get(UPPER_BODY_CONTROLLER, {}).get("ros__parameters", {})
        joints = params.get("joints")
        if not isinstance(joints, list) or not all(isinstance(j, str) for j in joints):
            raise ROSConfigError(
                f"S0 policy {self.openral_policy.name!r}: "
                f"{UPPER_BODY_CONTROLLER}.ros__parameters.joints must be a list of joint names."
            )
        return list(joints)


def s0_policy_path(robot_model: str, policy: str) -> Path:
    """Locate ``config/<robot_model>/policies/<policy>.yaml``.

    Order: ``$OPENRAL_UNITREE_CONFIG_DIR``, the installed ``openral_hal_unitree`` share (a
    sourced overlay), then this repository checkout. The first existing file wins; with
    none, the repo-checkout path is returned so ``load_s0_policy`` names it in its error.
    """
    rel = Path(robot_model) / "policies" / f"{policy}.yaml"
    roots: list[Path] = []
    env = os.environ.get("OPENRAL_UNITREE_CONFIG_DIR")
    if env:
        roots.append(Path(env))
    installed = _installed_config_root()
    if installed is not None:
        roots.append(installed)
    roots.append(_REPO_CONFIG)
    for root in roots:
        if (root / rel).is_file():
            return root / rel
    return _REPO_CONFIG / rel


def _installed_config_root() -> Path | None:
    """``share/openral_hal_unitree/config`` of a sourced overlay, or ``None`` without one."""
    try:
        from ament_index_python.packages import (  # noqa: PLC0415  # reason: ROS-only, optional
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(UNITREE_BRINGUP_PACKAGE)) / "config"
    except (PackageNotFoundError, ValueError):
        return None


def load_s0_policy(path: Path, *, require_ready: bool = True) -> S0PolicyFile:
    """Load and validate one policy file.

    Raises:
        ROSConfigError: Missing or malformed file; a ``placeholder`` when ``require_ready``.
    """
    if not path.is_file():
        raise ROSConfigError(f"S0 policy file {path} does not exist.")
    try:
        policy = S0PolicyFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as exc:
        raise ROSConfigError(f"S0 policy file {path} is malformed: {exc}") from exc
    meta = policy.openral_policy
    if require_ready and meta.status != "ready":
        raise ROSConfigError(
            f"S0 policy {meta.name!r} is `status: {meta.status}` — not verified against a "
            f"pinned export, so it will not be started. Fill in {path} and set `status: ready`."
        )
    if INFERENCE_CONTROLLER not in policy.controllers:
        raise ROSConfigError(f"{path}: `controllers` lacks {INFERENCE_CONTROLLER}.")
    policy.upper_body_joints  # noqa: B018  # reason: validates the joint list eagerly
    return policy


def resolve_leapp_config(
    policy: S0PolicyFile,
    override: str,
    package_share: Callable[[str], str] | None = None,
) -> Path:
    """The LEAPP config the inference controller loads.

    ``override`` (a path) wins; else ``share/<data_package>/data/<leapp_config>`` (upstream's
    rule, needs ``package_share``); else ``<bundle_dir>/<leapp_config>``.

    Raises:
        ROSConfigError: Nothing resolves, the data package is missing, or the file is absent.
    """
    meta = policy.openral_policy
    if override.strip():
        path = Path(override.strip()).expanduser().resolve()
    elif meta.data_package is not None:
        if package_share is None:
            raise ROSConfigError(
                f"S0 policy {meta.name!r} names data package {meta.data_package!r}, "
                "but no package index was given to resolve it."
            )
        try:
            share = package_share(meta.data_package)
        except Exception as exc:  # reason: ament's PackageNotFoundError, re-raised typed
            raise ROSConfigError(
                f"S0 policy {meta.name!r} needs the data package {meta.data_package!r} (its "
                "LEAPP export), which is not installed. Install it per the policy's upstream "
                "deploy tutorial, or pass inference_controller_config_path:=<yaml>."
            ) from exc
        path = Path(share) / "data" / meta.leapp_config
    elif meta.bundle_dir is not None:
        path = Path(meta.bundle_dir).expanduser() / meta.leapp_config
    else:
        raise ROSConfigError(
            f"S0 policy {meta.name!r} declares neither data_package nor bundle_dir."
        )
    if not path.is_file():
        raise ROSConfigError(
            f"S0 policy LEAPP config {path} does not exist"
            + (
                " — build it with `uv run python tools/build_gr00t_wbc_leapp.py`."
                if meta.bundle_dir is not None and not override.strip()
                else "."
            )
        )
    return path


def _inference_controller_types(doc: dict[str, Any]) -> list[Any]:
    """Every ``type`` a ROS params document gives ``inference_controller``.

    Both forms the spawner or ``controller_manager`` reads:
    ``<manager key>.ros__parameters.inference_controller.type`` and
    ``<key ending in inference_controller>.ros__parameters.type`` — namespaced and wildcard
    keys such as ``/**`` included.
    """
    found: list[Any] = []
    for key, block in doc.items():
        params = block.get("ros__parameters") if isinstance(block, dict) else None
        if not isinstance(params, dict):
            continue
        if str(key).rstrip("/").endswith(INFERENCE_CONTROLLER) and "type" in params:
            found.append(params["type"])
        nested = params.get(INFERENCE_CONTROLLER)
        if isinstance(nested, dict) and "type" in nested:
            found.append(nested["type"])
    return found


def require_watchdog_enforced(policy: S0PolicyFile, base_params: dict[str, Any]) -> None:
    """Refuse a policy whose stale-input watchdog would not be enforced.

    The pinned Jazzy ``isaac_ros_deploy`` (v4.6) has no ``topic_input_timeout_ms`` /
    ``topic_input_sources``: its ``InferenceController`` accepts them without a word and
    keeps running the policy on its last velocity command (issue #565). Only
    ``GuardedInferenceController`` enforces them, and only with a positive timeout. So the
    policy must set ``topic_input_timeout_ms > 0``, and every ``type`` that ``base_params``
    (``controller_manager.yaml``) or the policy file gives ``inference_controller`` — in the
    manager's block or the controller's own, under any key — must be the guarded one.

    Raises:
        ROSConfigError: No positive timeout, no type, or any type other than the guarded one.
    """
    name = policy.openral_policy.name
    inference = policy.controllers[INFERENCE_CONTROLLER].get("ros__parameters", {})
    timeout = inference.get("topic_input_timeout_ms")
    # A float: ROS types the parameter from the YAML literal, and the controller refuses an int.
    if not isinstance(timeout, float) or not timeout > 0:
        raise ROSConfigError(
            f"S0 policy {name!r}: {INFERENCE_CONTROLLER}.topic_input_timeout_ms is {timeout!r}; "
            "a real robot needs a positive stale-input timeout in ms, written as a float "
            "(500.0), or a dead OpenRAL side leaves it walking on its last velocity command."
        )
    types = _inference_controller_types(base_params) + _inference_controller_types(
        policy.controllers
    )
    wrong = [t for t in types if t != GUARDED_INFERENCE_CONTROLLER]
    if not types or wrong:
        raise ROSConfigError(
            f"S0 policy {name!r} sets {INFERENCE_CONTROLLER}.topic_input_timeout_ms, but "
            f"{INFERENCE_CONTROLLER} is typed {types!r}; only {GUARDED_INFERENCE_CONTROLLER!r} "
            "enforces it on the pinned isaac_ros_deploy — any other type ignores it and a stale "
            "velocity command keeps the robot walking."
        )


def policy_parameters(policy: S0PolicyFile, leapp_config: Path) -> dict[str, Any]:
    """The policy's ``controllers`` block (a copy) with ``config_path`` filled in."""
    params: dict[str, Any] = yaml.safe_load(yaml.safe_dump(policy.controllers))
    params[INFERENCE_CONTROLLER]["ros__parameters"]["config_path"] = str(leapp_config)
    return params

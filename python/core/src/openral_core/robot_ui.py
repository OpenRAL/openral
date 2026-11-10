"""Contracts between a robot and its dashboard UI (ADR-0129).

* **The robot manifest** (Interface A). The runtime emits it as an OTLP log record
  (``event.name = "openral.robot.manifest"``) at start and whenever
  ``/openral/skill_registry_changed`` fires; the dashboard caches it and serves it as
  ``GET /api/manifest``. The robot UI renders only the panels and actions it declares.
* **The robot enrolment** (Interface B). ``openral robot link <code>`` decodes the
  website's link code and writes everything but the tunnel token to
  ``~/.config/openral/robot.json``; the dashboard's remote listener validates
  openral.com tokens against it.

Example:
    >>> from openral_core import DeployRuntime, RobotDescription
    >>> robot = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    >>> manifest = build_robot_manifest(
    ...     robot,
    ...     DeployRuntime(enable_reasoner=True),
    ...     [],
    ...     bridge_on=False,
    ...     write_controls=False,
    ...     remote_control=False,
    ... )
    >>> [a.id for a in manifest.actions]
    ['estop', 'estop_reset', 'prompt']
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal  # Any: JSON Schema dicts are untyped by nature
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    EXECUTE_RSKILL_KINDS,
    DeployRuntime,
    EmbodimentKind,
    RobotCapabilities,
    RobotDescription,
    RSkillManifest,
    SensorModality,
    SensorSpec,
)

__all__ = [
    "ENROLMENT_PATH_ENV",
    "REMOTE_CONTROL_ENV",
    "TRUSTED_ISSUERS",
    "TRUSTED_ISSUERS_ENV",
    "WRITE_CONTROLS_ENV",
    "CameraPosition",
    "ManifestAction",
    "ManifestPanel",
    "RobotEnrolment",
    "RobotManifest",
    "build_robot_manifest",
    "camera_position",
    "enrolment_path",
    "parse_robot_link_code",
    "rskill_matches_robot",
]

WRITE_CONTROLS_ENV = "OPENRAL_DASHBOARD_WRITE_CONTROLS"
"""``=1`` puts ``skill:*`` actions in the manifest and lets the dashboard run them (ADR-0084)."""
REMOTE_CONTROL_ENV = "OPENRAL_DASHBOARD_REMOTE_CONTROL"
"""``=1`` allows reset, prompt and skills over the tunnel listener (E-STOP always is)."""
ENROLMENT_PATH_ENV = "OPENRAL_ROBOT_ENROLMENT"
"""Overrides where the robot enrolment lives (default ``~/.config/openral/robot.json``)."""
TRUSTED_ISSUERS: tuple[str, ...] = ("https://www.openral.com", "https://dev.openral.com")
"""The origins a link code's ``iss`` may name: openral.com's production and dev sites."""
TRUSTED_ISSUERS_ENV = "OPENRAL_TRUSTED_ISSUERS"
"""Comma-separated origins that REPLACE ``TRUSTED_ISSUERS`` (self-hosted issuer, tests).

An origin listed here may also serve the ``jwks`` of an issuer on another origin."""

PanelType = Literal[
    "identity",
    "camera",
    "joints",
    "map",
    "pointcloud",
    "reasoner",
    "rskill",
    "reward",
    "safety_status",
    "safety_ledger",
    "system",
    "metrics",
    "events",
    "world_state",
    "viz3d",
]

# Sensors whose frames the dashboard shows as camera tiles.
_IMAGE_MODALITIES = frozenset(
    {
        SensorModality.RGB,
        SensorModality.DEPTH,
        SensorModality.STEREO,
        SensorModality.IR,
        SensorModality.TACTILE_VISION,
    }
)

_PROMPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string", "minLength": 1}},
    "required": ["text"],
}


CameraPosition = Literal["left", "center", "right"]

_SIDE_TOKEN = re.compile(r"(?:^|[^a-z])(left|right)(?:[^a-z]|$)")
_HAND_TOKEN = re.compile(r"(?:^|[^a-z])(wrist|hand|gripper|eef)(?:[^a-z]|$)")


def camera_position(sensor: SensorSpec, robot: RobotDescription) -> CameraPosition:
    """Where a camera sits in the console's camera row (left to right).

    1. ``left`` / ``right`` when the sensor's name or ``frame_id`` says so
       (``left_wrist``, ``cam_right``).
    2. ``left`` for a camera that moves with the robot: its ``parent_frame`` (or,
       without one, its ``frame_id``) is a robot link other than ``base_frame``,
       or its name or ``frame_id`` says wrist, hand, gripper or eef.
    3. ``center`` otherwise (fixed to the world or the base, e.g. an overhead camera).

    Args:
        sensor: The camera's sensor spec.
        robot: The robot it belongs to.

    Returns:
        ``"left"``, ``"center"`` or ``"right"``.

    Example:
        >>> robot = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        >>> {s.name: camera_position(s, robot) for s in robot.sensors if s.name in ("top", "wrist")}
        {'top': 'center', 'wrist': 'left'}
    """
    for text in (sensor.name, sensor.frame_id):
        m = _SIDE_TOKEN.search(text.lower())
        if m:
            return "left" if m.group(1) == "left" else "right"
    links = {j.parent_link for j in robot.joints} | {j.child_link for j in robot.joints}
    parent = sensor.parent_frame or sensor.frame_id
    if parent and parent != robot.base_frame and parent in links:
        return "left"
    if any(_HAND_TOKEN.search(text.lower()) for text in (sensor.name, sensor.frame_id)):
        return "left"
    return "center"


class ManifestPanel(BaseModel):
    """One panel the robot UI draws.

    Attributes:
        id: Unique within the manifest (``cam:<sensor>`` for cameras).
        type: Which renderer; the UI ignores a type it does not know.
        title: Heading shown on the panel.
        source: The sensor name, for ``camera`` panels.
        position: Where a ``camera`` panel sits in the console's camera row
            (``left``, ``center`` or ``right``; see ``camera_position``), else ``None``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    type: PanelType
    title: str
    source: str | None = None
    position: CameraPosition | None = None


class ManifestAction(BaseModel):
    """One action the robot accepts on ``POST /api/actions/{id}``.

    Attributes:
        id: ``estop``, ``estop_reset``, ``prompt`` or ``skill:<rskill id>``.
        kind: What the dashboard does with it.
        label: Button text.
        min_role: Lowest openral.com role allowed to run it.
        remote: Allowed over the tunnel listener right now (``estop`` always is).
        confirm: A question the UI asks before sending, or ``None``.
        args_schema: JSON Schema of ``args``, or ``None`` for no arguments.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    kind: Literal["estop", "estop_reset", "prompt", "skill"]
    label: str
    min_role: Literal["viewer", "operator"]
    remote: bool
    confirm: str | None = None
    args_schema: dict[str, Any] | None = None


class RobotManifest(BaseModel):
    """What a robot shows and accepts (ADR-0129 Interface A).

    Attributes:
        schema_version: Interface version, ``1`` (``"schema"`` on the wire).
        robot_id: ``RobotDescription.name``.
        name: Display name.
        embodiment_kind: The robot's kinematic class.
        unit_serial: This unit's serial, when known.
        base_frame: The TF frame the 3D view follows (``RobotDescription.base_frame``).
        panels: Panels, in display order.
        actions: Actions, ``estop`` first.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schema")
    robot_id: str = Field(min_length=1)
    name: str
    embodiment_kind: EmbodimentKind
    unit_serial: str | None = None
    base_frame: str = "base_link"
    panels: list[ManifestPanel]
    actions: list[ManifestAction]


def rskill_matches_robot(
    skill: RSkillManifest,
    capabilities: RobotCapabilities,
    *,
    commercial_deployment: bool = False,
) -> bool:
    """Whether a robot offers ``skill`` as a dispatchable tool.

    The one predicate the reasoner palette (``build_tool_palette``) and the robot
    manifest share: not the scaffold template; ``kind`` in ``EXECUTE_RSKILL_KINDS``;
    an ``embodiment_tags`` overlap; every ``capabilities_required`` key a ``True``
    capability flag; and, on a commercial deployment, a commercial-use licence.

    Example:
        >>> from openral_core import RobotDescription, RSkillManifest
        >>> robot = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
        >>> nav = RSkillManifest.from_yaml("rskills/rskill-nav2-navigate-to-pose/rskill.yaml")
        >>> rskill_matches_robot(nav, robot.capabilities)
        False
    """
    if skill.is_scaffold_placeholder or skill.kind not in EXECUTE_RSKILL_KINDS:
        return False
    if not set(skill.embodiment_tags) & set(capabilities.embodiment_tags):
        return False
    flags = {field for field, value in capabilities.model_dump().items() if value is True}
    if not set(skill.capabilities_required or []) <= flags:
        return False
    return not (commercial_deployment and not skill.is_commercial_use_allowed)


def _skill_label(rskill_id: str) -> str:
    """``OpenRAL/rskill-moveit-multi-joints`` → ``moveit-multi-joints``."""
    return rskill_id.rsplit("/", 1)[-1].removeprefix("rskill-")


def build_robot_manifest(
    robot: RobotDescription,
    runtime: DeployRuntime,
    installed: Sequence[RSkillManifest],
    *,
    bridge_on: bool,
    write_controls: bool,
    remote_control: bool,
    commercial_deployment: bool = False,
    unit_serial: str | None = None,
) -> RobotManifest:
    """Build the manifest a deploy emits.

    Pass ``runtime`` with its toggles resolved: a ``None`` (auto) reads as off.

    Args:
        robot: The robot's description.
        runtime: The deploy's resolved ``DeployRuntime`` toggles.
        installed: Every installed rSkill; only those ``rskill_matches_robot``
            admits become ``skill:<id>`` actions.
        bridge_on: ``foxglove_bridge`` runs, so the 3D panel has data.
        write_controls: ``$OPENRAL_DASHBOARD_WRITE_CONTROLS=1``; skills are left out otherwise.
        remote_control: ``$OPENRAL_DASHBOARD_REMOTE_CONTROL=1``; sets ``remote`` on every
            action but ``estop`` (which is always remote).
        commercial_deployment: Drop skills whose licence forbids commercial use.
        unit_serial: This unit's serial (``resolve_unit_identity``).

    Returns:
        The manifest; actions are ``estop`` first, then the rest in id order.
    """
    panels = [ManifestPanel(id="identity", type="identity", title="Robot")]
    panels += [
        ManifestPanel(
            id=f"cam:{s.name}",
            type="camera",
            title=s.name,
            source=s.name,
            position=camera_position(s, robot),
        )
        for s in robot.sensors
        if s.modality in _IMAGE_MODALITIES
    ]
    if robot.joints or robot.end_effectors:
        panels.append(ManifestPanel(id="joints", type="joints", title="Joints"))
    optional: list[tuple[bool | None, PanelType, str]] = [
        (runtime.enable_slam, "map", "Map"),
        (runtime.enable_octomap, "pointcloud", "Point cloud"),
        (runtime.enable_reasoner, "reasoner", "Reasoner"),
        (True, "rskill", "rSkill"),
        (runtime.enable_reward_monitor, "reward", "Reward"),
        (True, "safety_status", "Safety status"),
        (True, "safety_ledger", "Safety ledger"),
        (True, "system", "System"),
        (True, "metrics", "Metrics"),
        (True, "events", "Events"),
        (True, "world_state", "World state"),
        (bridge_on, "viz3d", "3D"),
    ]
    panels += [ManifestPanel(id=t, type=t, title=title) for on, t, title in optional if on]

    actions = [
        ManifestAction(id="estop", kind="estop", label="E-STOP", min_role="viewer", remote=True),
        ManifestAction(
            id="estop_reset",
            kind="estop_reset",
            label="Reset e-stop",
            min_role="operator",
            remote=remote_control,
            confirm="Is the work cell clear?",
        ),
    ]
    if runtime.enable_reasoner:
        actions.append(
            ManifestAction(
                id="prompt",
                kind="prompt",
                label="Prompt",
                min_role="operator",
                remote=remote_control,
                args_schema=_PROMPT_SCHEMA,
            )
        )
    if write_controls:
        skills = {
            s.name: s
            for s in installed
            if rskill_matches_robot(
                s, robot.capabilities, commercial_deployment=commercial_deployment
            )
        }
        actions += [
            ManifestAction(
                id=f"skill:{name}",
                kind="skill",
                label=_skill_label(name),
                min_role="operator",
                remote=remote_control,
                args_schema=skill.goal_params_schema or {"type": "object", "properties": {}},
            )
            for name, skill in sorted(skills.items())
        ]
    return RobotManifest(
        robot_id=robot.name,
        name=robot.name,
        embodiment_kind=robot.embodiment_kind,
        unit_serial=unit_serial,
        base_frame=robot.base_frame,
        panels=panels,
        actions=actions,
    )


def _origin_url(value: str, *, allow_loopback_http: bool = False) -> str:
    """Refuse anything but an ``https`` URL (``http`` only on loopback, when allowed)."""
    parts = urlsplit(value)
    loopback = parts.hostname in {"127.0.0.1", "localhost", "::1"}
    if not parts.hostname or not (
        parts.scheme == "https" or (allow_loopback_http and parts.scheme == "http" and loopback)
    ):
        raise ValueError(f"{value!r} must be an https URL")
    return value


class RobotEnrolment(BaseModel):
    """This robot's identity on openral.com (``~/.config/openral/robot.json``).

    Attributes:
        v: Link-code version, ``1``.
        robot: The website's robot id (the token's ``robot`` claim).
        aud: This robot's own origin, ``https://r-<id>.openral.com`` (the token's ``aud``).
        org: The owning org's slug, ``""`` for a free robot.
        iss: The token issuer, ``https://www.openral.com/api/auth``; its origin must be
            in ``TRUSTED_ISSUERS`` (or ``$OPENRAL_TRUSTED_ISSUERS``, which replaces it).
        jwks: Where the issuer's signing keys are (``<iss>/jwks``), on ``iss``'s origin
            unless ``$OPENRAL_TRUSTED_ISSUERS`` lists its origin.
        serial: The unit's serial, entered when the robot was added on openral.com; a
            deploy uses it when neither ``$OPENRAL_UNIT_SERIAL`` nor the unit file sets one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    v: Literal[1]
    robot: str = Field(min_length=1)
    aud: str
    org: str
    iss: str
    jwks: str
    serial: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:-]{1,64}$")

    @field_validator("aud", "iss")
    @classmethod
    def _https(cls, value: str) -> str:
        return _origin_url(value)

    @field_validator("jwks")
    @classmethod
    def _https_or_loopback(cls, value: str) -> str:
        return _origin_url(value, allow_loopback_http=True)

    @model_validator(mode="after")
    def _trusted_issuer(self) -> RobotEnrolment:
        # The robot trusts tokens signed by `jwks` and derives its account-settings URL
        # from `iss`: a code naming any other site hands that site the robot.
        override = _trusted_issuers_override()
        trusted = override if override is not None else TRUSTED_ISSUERS
        iss, jwks = _origin(self.iss), _origin(self.jwks)
        if iss not in trusted:
            raise ValueError(
                f"iss {self.iss!r} is not a trusted issuer ({', '.join(trusted)}); "
                f"set {TRUSTED_ISSUERS_ENV} for a self-hosted one"
            )
        if jwks != iss and jwks not in (override or ()):
            raise ValueError(
                f"jwks {self.jwks!r} is not on the issuer's origin {iss}; "
                f"list it in {TRUSTED_ISSUERS_ENV} to allow that"
            )
        return self


def _origin(url: str) -> str:
    """``scheme://host[:port]`` lowercased, a default port (443 / 80) dropped (RFC 6454).

    Example:
        >>> _origin("https://WWW.openral.com:443/api/auth"), _origin("http://127.0.0.1:8080/x")
        ('https://www.openral.com', 'http://127.0.0.1:8080')
    """
    parts = urlsplit(url)
    scheme, host = parts.scheme.lower(), (parts.hostname or "").lower()
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    port = parts.port
    default = {"https": 443, "http": 80}.get(scheme)
    return f"{scheme}://{host}" if port in (None, default) else f"{scheme}://{host}:{port}"


def _trusted_issuers_override() -> tuple[str, ...] | None:
    raw = os.environ.get(TRUSTED_ISSUERS_ENV)
    if not raw:
        return None
    return tuple(_origin(o.strip()) for o in raw.split(",") if o.strip())


def enrolment_path() -> Path:
    """``$OPENRAL_ROBOT_ENROLMENT``, else ``~/.config/openral/robot.json``."""
    env = os.environ.get(ENROLMENT_PATH_ENV)
    return Path(env) if env else Path.home() / ".config" / "openral" / "robot.json"


def parse_robot_link_code(code: str) -> tuple[RobotEnrolment, str, str | None]:
    """Decode an ``openral robot link`` code into the enrolment, tunnel token and robot secret.

    The code is base64url (padding optional) of
    ``{"v", "robot", "aud", "org", "iss", "jwks", "serial", "secret", "tunnel"}``.
    ``secret`` is the robot's own credential for its openral.com account settings
    (``ors_…``); codes from before it carry none.

    Raises:
        ROSConfigError: Not base64url JSON, a field missing or malformed, no tunnel
            token, or an ``iss`` / ``jwks`` outside ``TRUSTED_ISSUERS`` (or
            ``$OPENRAL_TRUSTED_ISSUERS``; ``RobotEnrolment`` checks it, so a stored
            enrolment is refused the same way when loaded).

    Example:
        >>> import base64, json
        >>> raw = {
        ...     "v": 1,
        ...     "robot": "k3x9",
        ...     "aud": "https://r-k3x9.openral.com",
        ...     "org": "",
        ...     "iss": "https://www.openral.com/api/auth",
        ...     "jwks": "https://www.openral.com/api/auth/jwks",
        ...     "tunnel": "eyJh",
        ... }
        >>> code = base64.urlsafe_b64encode(json.dumps(raw).encode()).decode().rstrip("=")
        >>> parse_robot_link_code(code)[0].robot
        'k3x9'
    """
    try:
        raw = json.loads(base64.urlsafe_b64decode(code.strip() + "=" * (-len(code.strip()) % 4)))
    except (binascii.Error, ValueError) as exc:
        raise ROSConfigError(f"not an openral robot link code: {exc}") from exc
    if not isinstance(raw, dict):
        raise ROSConfigError("not an openral robot link code: not a JSON object")
    tunnel = raw.pop("tunnel", None)
    if not isinstance(tunnel, str) or not tunnel:
        raise ROSConfigError("the robot link code carries no tunnel token")
    # It lands in a systemd EnvironmentFile: a newline there would add variables.
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]+", tunnel):
        raise ROSConfigError("the robot link code carries a malformed tunnel token")
    secret = raw.pop("secret", None)
    if secret is not None and not (isinstance(secret, str) and secret.startswith("ors_")):
        raise ROSConfigError("the robot link code carries a malformed robot secret")
    try:
        return RobotEnrolment.model_validate(raw), tunnel, secret
    except ValidationError as exc:
        raise ROSConfigError(f"invalid robot link code: {exc}") from exc

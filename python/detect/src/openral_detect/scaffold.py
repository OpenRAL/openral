"""Scaffold a :class:`RobotEnvironment` deployment config from detection.

``openral detect`` already learns almost everything a hardware deployment
needs — the robot identity (``robot_id``), the serial transport (the matched
USB ``port``), every camera and its ``/dev/video*`` device path, and the robot
limits (which live in the robot's own ``robots/<name>/robot.yaml`` and are
applied automatically by ``openral deploy run``). The only things detection
*cannot* know are the **task** (what job the robot should do) and the **VLA**
(which policy drives it). This module turns the detected
:class:`RobotDescription` into a schema-valid :class:`RobotEnvironment` with the
known fields filled and the two unknowable fields left as obvious ``TODO``
placeholders for the operator to edit before ``openral deploy run``.

The boundary is deliberate (CLAUDE.md §1.2, §1.4): we never invent a task
instruction or a policy reference, because a wrong one would silently drive
real motors.
"""

from __future__ import annotations

from openral_core.schemas import (
    HalConfig,
    RobotDescription,
    RobotEnvironment,
    SensorReaderBackend,
    SensorReaderConfig,
    SensorSpec,
    TaskSpec,
    VLASpec,
)

from openral_detect.report import DetectionReport

__all__ = [
    "TODO_TASK_ID",
    "TODO_VLA_WEIGHTS_URI",
    "scaffold_robot_environment",
]

# Sentinel placeholder values. Self-documenting and schema-valid:
# ``TODO_VLA_WEIGHTS_URI`` carries no URI scheme, so it passes
# ``RobotEnvironment``'s "bare rSkill reference" guard while remaining an
# obvious edit-me marker. The CLI greps for these to warn before deploy.
TODO_TASK_ID = "TODO/set-task-id"
TODO_TASK_INSTRUCTION = "TODO: natural-language goal handed to the VLA"
TODO_VLA_ID = "TODO-set-vla-adapter-id"
TODO_VLA_WEIGHTS_URI = "TODO-set-rskill-reference"

# Default serial port when neither detection nor the manifest supplies one.
_FALLBACK_PORT = "/dev/ttyUSB0"


def scaffold_robot_environment(
    description: RobotDescription,
    detection: DetectionReport | None = None,
) -> RobotEnvironment:
    """Build a deployment :class:`RobotEnvironment` from a detected robot.

    Pre-fills everything detection knows — ``robot_id``, the HAL serial
    ``port``, and one :class:`SensorReaderConfig` per camera (with the probed
    ``/dev/video*`` device path when known) — and leaves ``task`` / ``vla`` as
    ``TODO`` placeholders the operator must edit. ``safety`` is left ``None`` so
    the robot's own :attr:`RobotDescription.safety` envelope (the robot limits)
    applies unchanged.

    Args:
        description: The assembled :class:`RobotDescription`, e.g. the output
            of :func:`openral_detect.assemble_robot_description`.
        detection: The originating :class:`DetectionReport`. Optional — when
            supplied, the matched USB device's ``port`` is used for the HAL
            transport; otherwise the manifest's ``hal.parameters.defaults`` (or
            ``/dev/ttyUSB0``) is used.

    Returns:
        A validated :class:`RobotEnvironment`. It loads and round-trips, but is
        **not runnable** until ``task`` and ``vla`` are filled in.

    Example:
        >>> from openral_core.schemas import RobotDescription
        >>> desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        >>> env = scaffold_robot_environment(desc)
        >>> env.robot_id
        'so101_follower'
        >>> env.vla.weights_uri
        'TODO-set-rskill-reference'
    """
    port = _resolve_port(description, detection)
    manifest_defaults = dict(description.hal.parameters.defaults)
    # ``port`` is promoted to transport; the remaining manifest defaults
    # (e.g. ``calibrate_on_connect``) carry through as HAL params verbatim.
    manifest_defaults.pop("port", None)

    hal = HalConfig(
        adapter=description.name,
        transport={"port": port},
        params=manifest_defaults,
    )

    sensors = _scaffold_sensors(description)

    return RobotEnvironment(
        robot_id=description.name,
        hal=hal,
        sensors=sensors,
        task=TaskSpec(
            id=TODO_TASK_ID,
            scene_id="real_world",
            instruction=TODO_TASK_INSTRUCTION,
        ),
        vla=VLASpec(id=TODO_VLA_ID, weights_uri=TODO_VLA_WEIGHTS_URI),
        safety=None,  # robot limits come from RobotDescription.safety
        metadata={
            "generated_by": "openral detect --deployment",
            "edit_before_deploy": ["task", "vla"],
        },
    )


def _resolve_port(description: RobotDescription, detection: DetectionReport | None) -> str:
    """Pick the serial port: detected USB match → manifest default → fallback."""
    if detection is not None:
        for match in detection.usb.matches:
            if match.device.port:
                return match.device.port
        for device in detection.usb.devices:
            if device.port:
                return device.port
    default = description.hal.parameters.defaults.get("port")
    if isinstance(default, str) and default:
        return default
    return _FALLBACK_PORT


def _scaffold_sensors(description: RobotDescription) -> list[SensorReaderConfig]:
    """One reader config per camera, with the probed device path when known.

    Covers both flat :attr:`RobotDescription.sensors` and the sensors inside
    each :attr:`RobotDescription.sensor_bundles` (e.g. a RealSense pair). A
    sensor whose ``metadata`` carries a ``device_path`` (added by
    ``openral detect``) gets an ``opencv_thread`` reader bound to that
    ``/dev/video*`` node; otherwise an empty-param reader is emitted for the
    operator to point at the right device.
    """
    configs: list[SensorReaderConfig] = []
    seen: set[str] = set()
    flat: list[SensorSpec] = list(description.sensors)
    for bundle in description.sensor_bundles:
        flat.extend(bundle.sensors)
    for spec in flat:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        device_path = spec.metadata.get("device_path")
        backend_params: dict[str, object] = {}
        if isinstance(device_path, str) and device_path:
            backend_params = {"device": device_path, "fps": int(spec.rate_hz)}
        configs.append(
            SensorReaderConfig(
                sensor_id=spec.name,
                backend=SensorReaderBackend.OPENCV_THREAD,
                backend_params=backend_params,
            )
        )
    return configs

"""Shared BEHAVIOR-1K R1Pro wire constants and option helpers.

Single source of truth for the OmniGibson evaluator's raw observation keys and
the official R1Pro state/action widths, imported by the scene backend
(``openral_sim.backends.behavior``), the policy adapter
(``openral_sim.policies.behavior_groot``), and the ``openral behavior
serve`` WebSocket bridge (``openral_cli.behavior``), which would otherwise
each hand-copy the constants and drift independently.

The out-of-process sidecar scripts under ``tools/`` run in isolated Python
environments that cannot import ``openral_sim``; their copies of these
constants are the wire-contract mirror and must be updated in lockstep with
this module.
"""

from __future__ import annotations

from openral_core.exceptions import ROSConfigError

# The official evaluator's proprioception key and the R1Pro contract widths.
STATE_KEY = "robot_r1::proprio"
STATE_DIM = 61
ACTION_DIM = 23

# Camera role → the evaluator's sensor name; the raw observation carries the
# RGB image under ``<sensor>::rgb``.
CAMERA_SENSORS: dict[str, str] = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
}
CAMERA_RGB_KEYS: dict[str, str] = {
    role: f"{sensor}::rgb" for role, sensor in CAMERA_SENSORS.items()
}

# The R1Pro modality split of the 61-D proprio (the GR00T sidecar's
# ``_register_r1pro_modality``), in the B1K checkpoint's state modality order:
# (group, start, end) slices.
R1PRO_STATE_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("base_qvel", 0, 3),
    ("torso", 53, 57),
    ("left_arm", 3, 10),
    ("left_gripper", 24, 26),
    ("right_arm", 28, 35),
    ("right_gripper", 49, 51),
)
# The absolute gripper dims of the 23-D action (base 3, torso 4, left arm 7,
# left gripper 1, right arm 7, right gripper 1).
GRIPPER_ACTION_INDICES: tuple[int, int] = (14, 22)


def explicit_port(
    env_value: str | None,
    opt_value: object,
    default: int,
    *,
    env_var: str,
) -> int:
    """Resolve a sidecar port: explicit values must parse or fail loud.

    A user pointing at an already-running sidecar via the env var or the scene
    YAML must never have a typo silently discarded in favour of the
    hash-derived default port (which ends in a spurious boot timeout or a
    duplicate sidecar with no hint the explicit port was ignored).

    Args:
        env_value: Raw environment-variable value (wins when set non-empty).
        opt_value: The scene/policy option value (used when the env var is
            unset); ``None`` means "not configured".
        default: The hash-derived default port.
        env_var: Environment-variable name, for the error message.

    Returns:
        The resolved port.

    Raises:
        ROSConfigError: an explicitly-provided value does not parse as an int.
    """
    source: tuple[str, object] | None = None
    if env_value is not None and env_value.strip() != "":
        source = (env_var, env_value)
    elif opt_value is not None:
        source = ("backend_options.port", opt_value)
    if source is None:
        return default
    label, raw = source
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ROSConfigError(
            f"Malformed explicit sidecar port {raw!r} from {label} — expected an "
            f"integer. Fix or unset it (the hash-derived default is {default})."
        ) from exc

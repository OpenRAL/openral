"""Single sim/real HAL construction seam (ADR-0031).

:func:`build_hal` is the one place that turns a :class:`RobotDescription` +
a ``mode`` into a constructed :class:`~openral_hal.protocol.HAL`. Every
caller — the ROS lifecycle nodes (``deploy sim``), the runner factory
(``deploy run``), and the deploy-sim CLI — routes through it, so the choice
of HAL *type* lives only in the manifest's ``hal:`` block, never in
environment config or runtime parameters.

Routing:

* ``mode="sim"`` + ``sim_env_yaml`` → a :class:`~openral_hal.sim_attached.SimAttachedHAL`
  wrapping the scene's :class:`~openral_sim.rollout.SimRollout` (ADR-0034). The scene
  owns physics + pixels; the bare-twin / ``hal.sim`` class is bypassed entirely.
* ``mode="sim"`` → the manifest's ``hal.sim`` import string, or — when that
  is ``None`` and a ``sim:`` block is present — the derived
  :meth:`MujocoArmHAL.from_description` (ADR-0023). No sim HAL and no ``sim:``
  block → :class:`ROSCapabilityMismatch`.
* ``mode="real"`` → the manifest's ``hal.real`` import string, constructed
  with the supplied ``transport`` kwargs (real HALs take transport-specific
  arguments — serial ``port``, ``robot_ip``, ``fci_ip`` — and embed their own
  description). ``hal.real`` is ``None`` → :class:`ROSCapabilityMismatch`
  (the robot is simulation-only).
* ``mode="real"`` + ``sim_env_yaml`` → :class:`ROSConfigError` (a real-hardware
  HAL never attaches a sim scene).

Construction convention: a class whose ``__init__`` accepts a ``description``
parameter (the ros2_control real HALs) receives it; otherwise the class
self-describes (the zero-arg MuJoCo sim subclasses, the lerobot followers)
and only the ``transport`` keys its signature accepts are passed.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Literal, cast

from openral_core import RobotDescription
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError

from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal.protocol import HAL

__all__ = ["build_hal"]

HalMode = Literal["sim", "real"]


def build_hal(
    description: RobotDescription,
    *,
    mode: HalMode,
    transport: dict[str, object] | None = None,
    sim_env_yaml: str | None = None,
) -> HAL:
    """Construct the simulation or real-hardware HAL for ``description``.

    Args:
        description: The robot manifest (typically loaded via
            :meth:`RobotDescription.from_yaml`).
        mode: ``"sim"`` for the simulation HAL (``deploy sim`` / ``sim run``
            harness), ``"real"`` for the real-hardware HAL (``deploy run``).
        transport: Constructor kwargs for the real HAL (serial ``port``,
            ``robot_ip``, ``fci_ip``, …). Keys the target constructor does not
            accept are dropped. Ignored by the derived sim path. Merged
            **over** the manifest's ``hal.parameters.defaults`` (ADR-0029), so
            an explicit ``deploy run`` transport override wins.
        sim_env_yaml: Path to a SimScene YAML (ADR-0034; renamed from
            SceneEnvironment in ADR-0041). When
            provided with ``mode="sim"``, returns a
            :class:`~openral_hal.sim_attached.SimAttachedHAL` wrapping the
            scene's :class:`~openral_sim.rollout.SimRollout`; bypasses the
            bare-twin / ``hal.sim`` class. Mutually exclusive with
            ``mode="real"`` — raises :class:`~openral_core.exceptions.ROSConfigError`
            if both are supplied.

    Returns:
        An un-connected HAL instance.

    Raises:
        ROSCapabilityMismatch: The robot has no HAL for ``mode`` (sim-only
            robot asked for ``"real"``, or real-only robot asked for
            ``"sim"`` with no ``sim:`` block to derive from).
        ROSConfigError: ``mode`` is invalid, ``sim_env_yaml`` is supplied
            with ``mode="real"``, or a declared entrypoint is malformed /
            unresolvable.

    Example:
        >>> from openral_core import RobotDescription
        >>> desc = RobotDescription.from_yaml("robots/so100_follower/robot.yaml")  # doctest: +SKIP
        >>> hal = build_hal(desc, mode="sim")  # doctest: +SKIP
    """
    # ADR-0029 — the manifest's hal.parameters.defaults supply the HAL's
    # construction kwargs (serial port, robot_ip, …) so a parameterised robot
    # needs no bespoke lifecycle subclass. Explicit ``transport`` overrides
    # them; _construct() then drops any key the constructor does not accept.
    resolved = {**description.hal.parameters.defaults, **(transport or {})}
    if sim_env_yaml is not None and mode != "sim":
        raise ROSConfigError(
            "build_hal: sim_env_yaml is only valid with mode='sim' "
            f"(got mode={mode!r}); a real-hardware HAL never attaches a sim scene."
        )
    if mode == "sim":
        if sim_env_yaml is not None:
            # ADR-0034 — deploy sim attaches the scene's SimRollout behind a
            # SimAttachedHAL (the scene owns physics + pixels). Bypasses the
            # bare twin / hal.sim class. openral_sim imported lazily so
            # openral_hal stays import-safe without the sim group.
            from openral_hal.sim_attached import SimAttachedHAL
            from openral_hal.sim_bringup import build_sim_env_from_yaml

            env, seed = build_sim_env_from_yaml(sim_env_yaml, robot_id_fallback=description.name)
            return SimAttachedHAL(env, description, env_reset_seed=seed)
        entry = description.hal.sim
        if entry is None:
            if description.sim is None:
                raise ROSCapabilityMismatch(
                    f"robot {description.name!r} has no simulation HAL: hal.sim is "
                    "null and there is no `sim:` block to derive MujocoArmHAL from. "
                    "It is real-hardware-only — use `deploy run`."
                )
            return MujocoArmHAL.from_description(description)
        return _construct(_import_object(entry), description, resolved)
    if mode == "real":
        entry = description.hal.real
        if entry is None:
            raise ROSCapabilityMismatch(
                f"robot {description.name!r} has no real-hardware HAL (hal.real is "
                "null); it is simulation-only — use `deploy sim` / `sim run`."
            )
        return _construct(_import_object(entry), description, resolved)
    raise ROSConfigError(f"build_hal: unknown mode {mode!r}; expected 'sim' or 'real'.")


def _construct(obj: object, description: RobotDescription, transport: dict[str, object]) -> HAL:
    """Instantiate ``obj``, threading ``description`` and/or ``transport``.

    Passes ``description`` only when the constructor names that parameter;
    filters ``transport`` to the constructor's accepted kwargs unless it
    declares ``**kwargs``.
    """
    if not callable(obj):
        raise ROSConfigError(f"HAL entrypoint resolved to a non-callable {obj!r}.")
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return cast(HAL, obj(**transport))
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = dict(transport) if has_var_kw else {k: v for k, v in transport.items() if k in params}
    if "description" in params and "description" not in kwargs:
        kwargs["description"] = description
    return cast(HAL, obj(**kwargs))


def _import_object(path: str) -> object:
    """Resolve a ``"module.path:Attribute"`` string to the referenced object.

    Raises:
        ROSConfigError: The string is malformed, the module is not importable,
            or the attribute is absent.
    """
    if ":" not in path:
        raise ROSConfigError(
            f"HAL entrypoint {path!r} is malformed; expected 'module.path:Attribute'."
        )
    module_path, _, attr = path.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ROSConfigError(
            f"HAL entrypoint {path!r}: module {module_path!r} is not importable ({exc})."
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ROSConfigError(
            f"HAL entrypoint {path!r}: module {module_path!r} has no attribute {attr!r}."
        ) from exc

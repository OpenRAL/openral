"""Registries that map ID strings to backend factories.

Three registries:
    * :data:`SCENES`   — ``scene_id`` -> :class:`~openral_sim.rollout.SimRollout` factory.
    * :data:`POLICIES` — ``vla_id``   -> :class:`~openral_sim.policy.PolicyAdapter` factory.
    * :data:`ROBOTS`   — ``robot_id`` -> :class:`~openral_core.RobotDescription` factory.

The factory functions are kept thin so the registry stays serialisation-friendly
(IDs are plain strings inside YAML configs).  Heavy backend imports happen inside
the factory bodies, NOT at registry-decoration time, so installing
``openral-sim`` never pulls LIBERO / MetaWorld / torch transitively.

Example::

    from openral_sim.registry import SCENES


    @SCENES.register("my_scene")
    def _build(env_cfg):
        from my_scene_pkg import MySim  # imported lazily

        return MySim(env_cfg)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_core import RobotDescription, SimEnvironment

    from openral_sim.policy import PolicyAdapter
    from openral_sim.rollout import SimRollout


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., object])


class _Registry(Generic[T]):
    """Tiny ID → factory map with friendly errors.

    The registry stores plain callables; the type variable ``T`` describes the
    object the callable returns, not the callable itself, so users get the
    expected type from :meth:`get` (``mypy --strict`` happy).

    Attributes:
        kind: Human-readable label used in error messages (``"scene"``, …).
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, Callable[..., T]] = {}
        self._fixed_robots: dict[str, str] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def register(
        self,
        name: str,
        *,
        fixed_robot: str | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to register a factory under ``name``.

        Args:
            name: Stable string ID used in YAML configs.
            fixed_robot: Only meaningful on the ``SCENES`` registry. When the
                underlying physics backend hard-wires a single robot (LIBERO
                always instantiates Franka, MetaWorld always instantiates
                Sawyer, etc.) the scene declares it here. The CLI then
                rejects any ``--robot`` / ``robot_id`` value that disagrees
                with a typed :class:`ROSConfigError`, instead of silently
                running the scene's hardcoded robot. Leave ``None`` for
                free-axis scenes (``mock``, ``maniskill3``, ``simpler_env``).

        Returns:
            A decorator that records the factory and returns it unchanged.

        Raises:
            ROSConfigError: If ``name`` is already registered.
        """

        def _decorator(fn: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise ROSConfigError(
                    f"{self._kind} id {name!r} is already registered to "
                    f"{self._items[name].__module__}.{self._items[name].__qualname__}"
                )
            self._items[name] = fn
            if fixed_robot is not None:
                self._fixed_robots[name] = fixed_robot
            return fn

        return _decorator

    def fixed_robot(self, name: str) -> str | None:
        """Return the scene's hard-fixed robot id, or ``None`` if free-axis.

        Returns ``None`` for unregistered ``name`` as well — callers that
        care about "does this scene exist" should use :meth:`get`.
        """
        return self._fixed_robots.get(name)

    def get(self, name: str) -> Callable[..., T]:
        """Look up a factory by ID.

        Raises:
            ROSConfigError: If ``name`` is unknown — the message lists the
                ids that ARE registered to make typos easy to fix.
        """
        try:
            return self._items[name]
        except KeyError as exc:
            known = sorted(self._items)
            raise ROSConfigError(
                f"unknown {self._kind} id {name!r}; registered ids: {known if known else '<none>'}"
            ) from exc

    def names(self) -> list[str]:
        """Return the sorted list of registered IDs."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items


SceneFactory = Callable[["SimEnvironment"], "SimRollout"]
PolicyFactory = Callable[["SimEnvironment"], "PolicyAdapter"]
RobotFactory = Callable[[], "RobotDescription"]

SCENES: _Registry[SimRollout] = _Registry("scene")
POLICIES: _Registry[PolicyAdapter] = _Registry("policy")
ROBOTS: _Registry[RobotDescription] = _Registry("robot")

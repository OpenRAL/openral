"""Registries that map ID strings to backend factories.

Three registries:
    * ``SCENES``   — ``scene_id`` -> ``SimRollout`` factory.
    * ``POLICIES`` — ``vla_id``   -> ``PolicyAdapter`` factory.
    * ``ROBOTS``   — ``robot_id`` -> ``RobotDescription`` factory.

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

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Generic, TypeVar

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_core import RobotDescription

    from openral_sim.policy import PolicyAdapter
    from openral_sim.rollout import SimRollout


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., object])


class _Registry(Generic[T]):
    """Tiny ID → factory map with friendly errors.

    The registry stores plain callables; the type variable ``T`` describes the
    object the callable returns, not the callable itself, so users get the
    expected type from ``get`` (``mypy --strict`` happy).

    Attributes:
        kind: Human-readable label used in error messages (``"scene"``, …).
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, Callable[..., T]] = {}
        self._fixed_robots: dict[str, frozenset[str]] = {}
        self._provisioners: dict[str, Callable[[], None]] = {}
        self._meta: dict[str, dict[str, object]] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def register(
        self,
        name: str,
        *,
        fixed_robot: str | frozenset[str] | None = None,
        provision: Callable[[], None] | None = None,
        **meta: object,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to register a factory under ``name``.

        Args:
            name: Stable string ID used in YAML configs. Ids are
                ``/``-namespaced: registering a family prefix (``robocasa``)
                makes every ``robocasa/<task>`` id resolve to it, so a new
                task is YAML, not a registration (longest prefix wins).
            fixed_robot: Only meaningful on the ``SCENES`` registry. The
                robot id(s) the backend's physics can actually instantiate
                (LIBERO always builds Franka, RoboCasa kitchens build
                ``panda_mobile`` or its VSLAM variant, …). ``resolve_robot``
                enforces it at every entry point (``sim run``, ``benchmark``,
                ``deploy sim``, the sim HAL) with a typed ``ROSConfigError``
                instead of silently running a different robot. Leave ``None``
                for free-axis scenes (``mock``, ``tabletop_push``, ``isaac_sim``).
            provision: Only meaningful on the ``SCENES`` registry. Declares
                the scene's out-of-tree provisioning step — the multi-GB
                asset download, fork clone, or sidecar-venv build that the
                factory would otherwise trigger on its first call. Callers
                that can afford to do slow work up front (``openral deploy
                sim`` before ``ros2 launch``) run it via ``provision``
                so it does not land inside the HAL's ``on_configure``, a
                callback ``tools/lifecycle_autostart.py`` bounds at 300 s.
                Must be idempotent — the factory calls the same helpers
                again and they short-circuit on their own sentinels. Leave
                ``None`` for backends whose only setup is a pip install.
            **meta: Static facts about the entry, read back via ``meta``.
                On ``POLICIES`` these are the family's dependency facts
                (``install_groups``, ``required_imports``, optional
                ``install_note``) consumed by ``openral_sim.policy_deps``, so
                they live next to the factory instead of in parallel dicts.

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
                self._fixed_robots[name] = (
                    frozenset({fixed_robot}) if isinstance(fixed_robot, str) else fixed_robot
                )
            if provision is not None:
                self._provisioners[name] = provision
            self._meta[name] = dict(meta)
            return fn

        return _decorator

    def _key(self, name: str) -> str | None:
        """Resolve ``name`` to its registered key: exact, else longest ``/``-prefix.

        ``robocasa/gr1/PnPCupToDrawerClose`` -> ``robocasa/gr1`` ->
        ``robocasa``: a backend family registers once and the task stays
        data in the id. ``None`` when no prefix is registered.
        """
        key = name
        while key not in self._items and "/" in key:
            key = key.rsplit("/", 1)[0]
        return key if key in self._items else None

    def allowed_robots(self, name: str) -> frozenset[str] | None:
        """Return the robot ids the scene can instantiate, or ``None`` if free-axis.

        Returns ``None`` for unregistered ``name`` as well — callers that
        care about "does this scene exist" should use ``get``.
        """
        key = self._key(name)
        return None if key is None else self._fixed_robots.get(key)

    def fixed_robot(self, name: str) -> str | None:
        """Return the scene's default robot id, or ``None`` if free-axis.

        The default is what ``resolve_robot(name, None)`` picks: the only
        allowed robot, or the lexicographically first of several.
        """
        allowed = self.allowed_robots(name)
        return None if allowed is None else sorted(allowed)[0]

    def resolve_robot(self, name: str, requested: str | None) -> str:
        """Bind a robot to scene ``name`` — the one rule every entry point uses.

        Args:
            name: Scene id.
            requested: Robot id from ``--robot`` / the YAML's ``robot_id``,
                or ``None`` to take the scene's default.

        Returns:
            ``requested`` when the scene allows it (free-axis scenes allow
            any), else the scene's default robot when ``requested`` is None.

        Raises:
            ROSConfigError: ``requested`` is not among the scene's allowed
                robots, or the scene is free-axis and nothing was requested.

        Example:
            >>> from openral_sim import SCENES
            >>> SCENES.resolve_robot("robocasa/PickPlaceCounterToSink", None)
            'panda_mobile'
        """
        allowed = self.allowed_robots(name)
        if allowed is None:
            if requested is None:
                raise ROSConfigError(
                    f"{self._kind} {name!r} does not fix a robot; set `robot_id:` in "
                    "the YAML or pass --robot <robot_id>."
                )
            return requested
        if requested is None:
            return sorted(allowed)[0]
        if requested not in allowed:
            raise ROSConfigError(
                f"{self._kind} {name!r} can only instantiate {sorted(allowed)}; "
                f"robot {requested!r} is not one of them."
            )
        return requested

    def provision(self, name: str) -> Callable[[], None] | None:
        """Return the scene's pre-launch provisioner, or ``None``.

        ``None`` means "nothing slow to do ahead of time" — either the
        backend needs no out-of-tree assets, or ``name`` is not registered
        at all. Both are a no-op for the caller, so this deliberately does
        not raise on an unknown id the way ``get`` does: a preflight is
        advisory and the scene resolver owns reporting a bad scene id.
        """
        key = self._key(name)
        return None if key is None else self._provisioners.get(key)

    def meta(self, name: str) -> dict[str, object]:
        """Return the static facts registered with ``name`` (``{}`` if unknown).

        Like ``fixed_robot``, an unknown ``name`` is not an error here —
        callers that must know whether the id exists use ``get`` / ``in``.
        """
        key = self._key(name)
        return {} if key is None else self._meta.get(key, {})

    def get(self, name: str) -> Callable[..., T]:
        """Look up a factory by ID.

        Raises:
            ROSConfigError: If ``name`` is unknown — the message lists the
                ids that ARE registered to make typos easy to fix.
        """
        key = self._key(name)
        if key is None:
            known = sorted(self._items)
            raise ROSConfigError(
                f"unknown {self._kind} id {name!r}; registered ids: {known if known else '<none>'}"
            )
        return self._items[key]

    def names(self) -> list[str]:
        """Return the sorted list of registered IDs."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._key(name) is not None

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered IDs, so ``sorted(REGISTRY)`` works in error messages."""
        return iter(self._items)


SCENES: _Registry[SimRollout] = _Registry("scene")
POLICIES: _Registry[PolicyAdapter] = _Registry("policy")
ROBOTS: _Registry[RobotDescription] = _Registry("robot")

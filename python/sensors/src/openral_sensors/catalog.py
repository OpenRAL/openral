"""Sensor catalog — vendor-agnostic registry of `SensorSpec` / `SensorBundle` factories.

Mirrors the role that ``rSkillManifest`` plays for skills: a typed, addressable
descriptor that decouples the sensor *identity* (``intel/realsense_d435i``) from
its *materialisation* (a Pydantic ``SensorSpec`` or ``SensorBundle`` with
nominal data-sheet values).

Entries are registered on import by each vendor module (``realsense.py``,
``orbbec.py``, ``slamtec.py``, …). The registry is consumed by:
- ``openral sensor list``  — print all registered ids.
- ``openral sensor show``  — pretty-print one resolved spec/bundle.
- The future ``catalog:`` reference field on ``RobotDescription.sensors`` /
  ``RobotDescription.sensor_bundles`` (gated on schema v0.3 ADR).

Design notes
------------
- A single global ``CATALOG`` instance is exposed.  Tests that need isolation
  use ``SensorCatalog()`` directly.
- Re-registration of an existing id raises by default; pass
  ``replace=True`` to overwrite (useful in tests).
- Factories are *not* called at registration time — the catalog stores the
  callable, so listing the catalog has zero side-effects.

Example:
    >>> from openral_sensors import CATALOG
    >>> "intel/realsense_d435" in CATALOG
    True
    >>> entry = CATALOG.get("intel/realsense_d435")
    >>> entry.kind
    'bundle'
    >>> bundle = CATALOG.build("intel/realsense_d435", name="head", parent_frame="base_link")
    >>> bundle.bundle_name
    'head'
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from openral_core.schemas import SensorBundle, SensorModality, SensorSpec

__all__ = [
    "CATALOG",
    "SensorCatalog",
    "SensorCatalogEntry",
    "SensorSignature",
]


SensorFactory = Callable[..., SensorSpec | SensorBundle]

SensorSignatureKind = Literal["realsense", "orbbec", "usb_uvc", "v4l2_name"]


@dataclass(frozen=True)
class SensorSignature:
    """A probe-side identifier that resolves to one catalog entry.

    ``SensorSignature`` is the bridge between a live device discovered by
    ``openral detect`` and the canonical ``SensorCatalogEntry`` whose factory
    materialises a fully-populated ``SensorSpec`` (with real intrinsics,
    FOV, encoding, rate, …).  Each catalog entry may publish multiple
    signatures (e.g. a USB UVC camera with several VID/PID rebrands).

    Encoding by ``kind``:
    - ``"realsense"`` — pyrealsense2 ``model_id`` (e.g. ``"D435I"``).
    - ``"orbbec"`` — Orbbec SDK product name.
    - ``"usb_uvc"`` — ``"0xVVVV:0xPPPP"`` lowercase hex.
    - ``"v4l2_name"`` — V4L2 product string (substring match acceptable).

    Attributes:
        kind: Probe family that produced this signature.
        value: Vendor-specific canonical key.

    Example:
        >>> SensorSignature(kind="realsense", value="D435I")
        SensorSignature(kind='realsense', value='D435I')
    """

    kind: SensorSignatureKind
    value: str


@dataclass(frozen=True)
class SensorCatalogEntry:
    """One row in the sensor catalog.

    Attributes:
        id: Stable identifier in the form ``"<vendor>/<model>"`` — lowercase,
            slugified.  Used as the lookup key and as the value of the future
            ``catalog:`` field in ``robot.yaml``.
        vendor: Vendor or maintainer name (free-form, lowercase).
        model: Vendor model identifier (free-form, lowercase / slugified).
        kind: Whether the factory returns a single ``SensorSpec`` or a
            multi-sensor ``SensorBundle``.
        factory: Callable that constructs the spec/bundle.  Must accept at
            least the same keyword arguments as the underlying factory; the
            common convention is ``name``, ``parent_frame``, ``rate_hz`` style.
        modalities: ``SensorModality`` values that this sensor exposes.
        description: One-line human-readable summary.
        docs_url: Optional link to the vendor data sheet or product page.
        signatures: Probe-side identifiers that resolve to this entry.  Used
            by ``openral detect`` to map a discovered device (RealSense ``model_id``,
            USB VID/PID, V4L2 product string) to the canonical catalog ID.
    """

    id: str
    vendor: str
    model: str
    kind: Literal["sensor", "bundle"]
    factory: SensorFactory
    modalities: tuple[SensorModality, ...]
    description: str = ""
    docs_url: str | None = None
    signatures: tuple[SensorSignature, ...] = ()


@dataclass
class SensorCatalog:
    """In-memory registry of ``SensorCatalogEntry`` rows.

    The default global instance is :data:`CATALOG`; tests should create a
    private :class:`SensorCatalog` to avoid polluting the global registry.

    Example:
        >>> cat = SensorCatalog()
        >>> def make() -> SensorSpec:
        ...     return SensorSpec(name="x", modality=SensorModality.RGB, frame_id="x", rate_hz=30.0)
        >>> _ = cat.register(
        ...     SensorCatalogEntry(
        ...         id="acme/cam",
        ...         vendor="acme",
        ...         model="cam",
        ...         kind="sensor",
        ...         factory=make,
        ...         modalities=(SensorModality.RGB,),
        ...     )
        ... )
        >>> cat.get("acme/cam").vendor
        'acme'
        >>> cat.list_ids()
        ['acme/cam']
    """

    _entries: dict[str, SensorCatalogEntry] = field(default_factory=dict)

    # ── Mutators ──────────────────────────────────────────────────────────────

    def register(self, entry: SensorCatalogEntry, *, replace: bool = False) -> SensorCatalogEntry:
        """Register an entry.

        Raises ``KeyError`` on duplicate id unless ``replace=True``.
        """
        if not replace and entry.id in self._entries:
            raise KeyError(
                f"Sensor catalog already has an entry for id={entry.id!r}; "
                "pass replace=True to overwrite."
            )
        self._entries[entry.id] = entry
        return entry

    def register_many(self, entries: Iterable[SensorCatalogEntry], *, replace: bool = True) -> None:
        """Bulk-register catalog entries.

        Used by vendor modules at import time to populate the global CATALOG.
        Defaults to ``replace=True`` because side-effect imports may run more
        than once in some test setups.
        """
        for entry in entries:
            self.register(entry, replace=replace)

    def unregister(self, sensor_id: str) -> None:
        """Remove an entry.  Idempotent."""
        self._entries.pop(sensor_id, None)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get(self, sensor_id: str) -> SensorCatalogEntry:
        """Look up an entry by id.  Raises ``KeyError`` if absent."""
        try:
            return self._entries[sensor_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown sensor id {sensor_id!r}.  Known ids: {sorted(self._entries)!r}"
            ) from exc

    def __contains__(self, sensor_id: object) -> bool:
        """Membership check by sensor id."""
        return isinstance(sensor_id, str) and sensor_id in self._entries

    def __len__(self) -> int:
        """Number of registered entries."""
        return len(self._entries)

    def __iter__(self) -> object:
        """Iterate over registered ids."""
        return iter(self._entries)

    def keys(self) -> list[str]:
        """All registered ids (insertion order)."""
        return list(self._entries.keys())

    def list_ids(self) -> list[str]:
        """All registered ids, sorted alphabetically."""
        return sorted(self._entries)

    def entries(self) -> list[SensorCatalogEntry]:
        """All entries, sorted alphabetically by id."""
        return [self._entries[k] for k in sorted(self._entries)]

    def filter(
        self,
        *,
        vendor: str | None = None,
        modality: SensorModality | None = None,
        kind: Literal["sensor", "bundle"] | None = None,
    ) -> list[SensorCatalogEntry]:
        """Return entries that match all provided predicates."""
        out: list[SensorCatalogEntry] = []
        for entry in self.entries():
            if vendor is not None and entry.vendor != vendor:
                continue
            if modality is not None and modality not in entry.modalities:
                continue
            if kind is not None and entry.kind != kind:
                continue
            out.append(entry)
        return out

    def find_by_signature(self, signature: SensorSignature) -> SensorCatalogEntry | None:
        """Look up an entry by probe-side signature.

        Returns ``None`` when no registered entry advertises the signature.
        Callers (the ``openral detect`` assembler) decide between catalog-backed
        materialisation and a fallback generic ``SensorSpec``.

        Example:
            >>> from openral_sensors import CATALOG
            >>> from openral_sensors.catalog import SensorSignature
            >>> entry = CATALOG.find_by_signature(SensorSignature(kind="realsense", value="D435I"))
            >>> entry is not None and entry.id == "intel/realsense_d435i"
            True
        """
        for entry in self._entries.values():
            if signature in entry.signatures:
                return entry
        return None

    # ── Construction ──────────────────────────────────────────────────────────

    def build(self, sensor_id: str, **kwargs: object) -> SensorSpec | SensorBundle:
        """Resolve ``sensor_id`` and call its factory with ``**kwargs``.

        This is the entry point used by the (future) ``catalog:`` reference in
        ``robot.yaml`` — the loader passes any per-instance overrides
        (``name``, ``parent_frame``, ``serial_no``, …) as keyword arguments.
        """
        entry = self.get(sensor_id)
        return entry.factory(**kwargs)


# Module-level singleton populated by vendor modules on import.
CATALOG = SensorCatalog()

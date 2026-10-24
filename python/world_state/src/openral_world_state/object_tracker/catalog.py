"""Catalog association and necessary-condition size mismatch checks; no tightening."""

from __future__ import annotations

import math
from dataclasses import dataclass

from openral_core import (
    BoxShape,
    CapsuleShape,
    ROSConfigError,
    TrackedObjectSet,
    USDDeclaredObject,
    USDObjectCatalog,
)


@dataclass
class _Association:
    epoch: int
    declaration: USDDeclaredObject
    active: bool = True
    reason: str = "catalog_unverified"
    last_stamp_ns: int = -1
    mismatch_stamp_ns: int = -1
    mismatch_span_m: float = 0.0
    mismatch_limit_m: float = 0.0


class CatalogDeclarations:
    """Bounded unique-label candidates with generation-scoped latched contradictions.

    A label association is a candidate, not identity proof. Partial surface spans
    can disprove a maximum dimension; they cannot verify containment or missing
    geometry. Nothing here supplies safety geometry or contact permission.
    """

    def __init__(self, catalog: USDObjectCatalog) -> None:
        """Keep the validated catalog and no track associations."""
        self.catalog = catalog
        self._states: dict[str, _Association] = {}

    def retain(self, identities: set[str]) -> None:
        """Remove retired track identities, preserving live mismatch latches."""
        self._states = {key: value for key, value in self._states.items() if key in identities}

    def associate(
        self, object_id: str, epoch: int, label: str, *, observed_label_count: int
    ) -> str:
        """Associate only one declared candidate with one observed label instance.

        Loss of uniqueness suspends the association without erasing its mismatch.
        An older lifecycle generation cannot overwrite a newer one.
        """
        old = self._states.get(object_id)
        if epoch < 0 or observed_label_count < 0:
            raise ROSConfigError("catalog association requires nonnegative epoch and count")
        if old is not None and epoch < old.epoch:
            return "catalog_stale_generation"
        if old is not None:
            old.epoch = epoch
        candidates = [
            obj for obj in self.catalog.objects if obj.label.casefold() == label.strip().casefold()
        ]
        if len(candidates) != 1 or observed_label_count != 1:
            if old is not None:
                old.active = False
            return "catalog_ambiguous" if candidates else "catalog_not_declared"
        declaration = candidates[0]
        if old is not None:
            if old.declaration.name != declaration.name:
                old.active = False
                return "catalog_identity_changed"
            old.epoch = epoch
            old.active = True
            return old.reason
        if old is None and len(self._states) >= TrackedObjectSet.MAX_OBJECTS:
            return "catalog_capacity"
        self._states[object_id] = _Association(epoch, declaration)
        return "catalog_unverified"

    def observe_span(
        self,
        object_id: str,
        epoch: int,
        measured_span_m: float,
        *,
        camera: str,
        stamp_ns: int,
        now_ns: int,
    ) -> str:
        """Check an observed lower bound on diameter with a calibrated error bound.

        Call only for uniquely attributed object points. Axis span is a valid
        lower bound; a containing-fit box diagonal is not. Calibration must cover
        depth, extrinsics and attribution, not merely registration residuals.
        Compatible partial observations never clear an oversize mismatch. A
        complete calibrated upper bound is required for future revalidation.
        """
        if not math.isfinite(measured_span_m) or measured_span_m < 0:
            raise ROSConfigError("catalog measured span must be finite and nonnegative")
        state = self._states.get(object_id)
        if state is None or state.epoch != epoch or not state.active:
            return "catalog_unassociated"
        calibration = self.catalog.measurement_calibration
        if calibration is None:
            return "catalog_uncalibrated"
        if camera not in calibration.sensor_ids:
            return "catalog_unsupported_sensor"
        if (
            stamp_ns <= 0
            or now_ns < stamp_ns
            or now_ns - stamp_ns > calibration.max_age_s * 1e9
            or stamp_ns <= state.last_stamp_ns
        ):
            return "catalog_stale_observation"
        primitive = state.declaration.shape
        if primitive is None or primitive.dimension_semantics != "maximum":
            return "catalog_no_maximum"
        shape = primitive.shape
        maximum = (
            2 * math.sqrt(sum(value * value for value in shape.half_extents_m))
            if isinstance(shape, BoxShape)
            else shape.length_m + 2 * shape.radius_m
            if isinstance(shape, CapsuleShape)
            else 2 * shape.radius_m
        )
        state.last_stamp_ns = stamp_ns
        if measured_span_m > maximum + 2 * calibration.point_error_bound_m:
            state.mismatch_span_m = measured_span_m
            state.mismatch_limit_m = maximum + 2 * calibration.point_error_bound_m
            state.mismatch_stamp_ns = stamp_ns
            state.reason = "catalog_dimension_mismatch"
        elif state.mismatch_stamp_ns < 0:
            state.reason = "catalog_span_compatible"
            state.mismatch_stamp_ns = -1
        return state.reason

    def blocked(self, object_id: str, epoch: int) -> bool:
        """Whether this or a later lifecycle generation has a latched contradiction."""
        state = self._states.get(object_id)
        return bool(state is not None and state.epoch <= epoch and state.mismatch_stamp_ns >= 0)

    def diagnostics(self) -> dict[str, str]:
        """Report bounded declaration provenance and latched state per identity."""
        calibration = self.catalog.measurement_calibration
        reference = calibration.calibration_reference if calibration else "none"
        return {
            key: (
                f"{state.reason}; span_m={state.mismatch_span_m:.6g}; "
                f"limit_m={state.mismatch_limit_m:.6g}; "
                f"declaration={state.declaration.name}; epoch={state.epoch}; "
                f"revision={self.catalog.source_revision}; "
                f"calibration={reference}"
            )
            for key, state in self._states.items()
        }

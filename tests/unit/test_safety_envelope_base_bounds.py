"""``SafetyEnvelope.max_base_*`` must be finite and positive (issue #405).

The kernel reads ``+inf`` as "no bound", so a manifest declaring it would pass the
cmd_vel relay's "both bounds set" refusal while leaving BODY_TWIST unbounded.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from openral_core import RobotDescription
from pydantic import ValidationError

_R1PRO = Path(__file__).resolve().parents[2] / "robots" / "r1pro" / "robot.yaml"


@pytest.mark.parametrize("field", ["max_base_linear_speed_m_s", "max_base_angular_speed_rad_s"])
@pytest.mark.parametrize("value", [math.inf, math.nan, 0.0, -0.3])
def test_base_speed_bound_rejects_non_finite_or_non_positive(field: str, value: float) -> None:
    safety = RobotDescription.from_yaml(str(_R1PRO)).safety
    with pytest.raises(ValidationError):
        type(safety).model_validate({**safety.model_dump(), field: value})


def test_real_manifest_bounds_still_validate() -> None:
    safety = RobotDescription.from_yaml(str(_R1PRO)).safety
    assert (safety.max_base_linear_speed_m_s, safety.max_base_angular_speed_rad_s) == (0.7, 0.3)

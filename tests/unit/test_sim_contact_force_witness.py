"""The sim contact-force producer, and the calibration it refuses to assume.

ADR-0100 (survey Path C) adds the second observable on the place path. At the
instant of a place the true clearance *is* zero, so no geometric margin at any
resolution separates "set the cup down on the shelf" from "crush it against the
shelf" (survey §9 point 4). Force is the axis that does.

Two things are pinned here, and the second matters more than the first.

1. The producer reads a real ``mj_contactForce`` off a real compiled ``MjModel``
   with a resting contact whose normal load is known analytically from the
   payload's own weight, and reports it against the declared target only.

2. **It refuses to call that number newtons.** No published work validates
   MuJoCo contact-force magnitudes against real force-torque measurements
   (survey §21.7); MuJoCo documents its contact model as an approximation
   resting on ``solref`` / ``solimp`` choices, and FORGE (arXiv:2408.04587)
   re-tunes its threshold on hardware across more than a thousand real trials.
   So a stock deployment publishes ``magnitude_calibrated=False`` and the kernel
   declines to read the magnitude at all. Arming it takes two explicit
   environment variables, one of which is a *name* for the calibration, because
   a scale nobody can audit is exactly the silent arming this design refuses.

No test here asserts that a MuJoCo magnitude equals real newtons. That blocker
is recorded, not closed.
"""

from __future__ import annotations

import numpy as np
import pytest
from openral_core import AttachmentEvidenceKind
from openral_hal._sim_attachment_evidence import (
    contact_force_calibration,
    probe_contact_force,
)

mujoco = pytest.importorskip("mujoco")

# The payload rests on the shelf under gravity, so the steady-state normal load
# is its own weight: m * g. Both are declared here so the assertion below is an
# analytic expectation rather than a recorded number.
_PAYLOAD_MASS_KG = 0.4
_GRAVITY_M_S2 = 9.81
_EXPECTED_NORMAL_LOAD = _PAYLOAD_MASS_KG * _GRAVITY_M_S2

_MJCF = f"""
<mujoco model="contact_force_witness">
  <option gravity="0 0 -{_GRAVITY_M_S2}"/>
  <worldbody>
    <body name="cabinet" pos="0 0 0">
      <body name="cabinet_shelf" pos="0 0 0">
        <geom name="cabinet_shelf_top" type="box" pos="0 0 0.38" size="0.16 0.25 0.02"
              contype="1" conaffinity="1"/>
      </body>
    </body>
    <body name="sideboard" pos="1.0 0 0.38">
      <geom name="sideboard_top" type="box" size="0.16 0.25 0.02" contype="1" conaffinity="1"/>
    </body>
    <body name="cup" pos="0 0 0.44">
      <freejoint name="cup_joint"/>
      <geom name="cup_body" type="box" size="0.03 0.03 0.03" mass="{_PAYLOAD_MASS_KG}"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _settled() -> tuple[object, object]:
    """A compiled model stepped until the payload rests on the shelf."""
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    for _ in range(600):
        mujoco.mj_step(model, data)
    return model, data


def _ids(model: object) -> tuple[list[int], int, int]:
    cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_body")
    shelf_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cabinet")
    sideboard = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sideboard")
    return [cup], shelf_root, sideboard


def test_a_resting_payload_attests_its_own_weight_against_the_declared_target() -> None:
    """The producer measures the contact, and names the target it is against."""
    model, data = _settled()
    payload, shelf_root, _ = _ids(model)

    witness = probe_contact_force(
        model,
        data,
        payload_geoms=payload,
        target_root_body=shelf_root,
        target_name="cabinet",
        stamp_ns=7,
    )

    assert witness is not None, "a payload resting on the shelf is in the solver's contact list"
    assert witness.target_id == "sim:cabinet"
    assert witness.evidence_kind is AttachmentEvidenceKind.SIM_CONTACT_FORCE
    assert witness.stamp_ns == 7
    # The normal load is the payload's own weight, which is what a settled
    # resting contact must carry. Loose tolerance: this is a solver at rest, not
    # an analytic identity, and tightening it would be asserting a precision the
    # contact model does not claim.
    assert witness.magnitude_n == pytest.approx(_EXPECTED_NORMAL_LOAD, rel=0.15)
    # Direction is a unit vector in the payload's own frame, pointing up out of
    # the shelf and into the cup.
    assert np.linalg.norm(witness.direction_in_object) == pytest.approx(1.0, abs=1e-9)
    assert witness.direction_in_object[2] > 0.9


def test_the_magnitude_is_not_claimed_to_be_newtons_without_a_named_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Survey §21.7, enforced at the producer rather than argued in a comment."""
    model, data = _settled()
    payload, shelf_root, _ = _ids(model)
    monkeypatch.delenv("OPENRAL_SIM_CONTACT_FORCE_N_PER_UNIT", raising=False)
    monkeypatch.delenv("OPENRAL_SIM_CONTACT_FORCE_CALIBRATION_REF", raising=False)

    stock = probe_contact_force(
        model,
        data,
        payload_geoms=payload,
        target_root_body=shelf_root,
        target_name="cabinet",
        stamp_ns=1,
    )
    assert stock is not None
    assert stock.magnitude_calibrated is False, "a stock sim arms no enforcement surface"
    assert stock.calibration_ref is None

    # A scale WITHOUT a name buys nothing: an unauditable magnitude is the silent
    # arming this refuses.
    monkeypatch.setenv("OPENRAL_SIM_CONTACT_FORCE_N_PER_UNIT", "2.0")
    scale, calibrated, reference = contact_force_calibration()
    assert (scale, calibrated, reference) == (1.0, False, None)

    # Both together, and only then, is the magnitude carried as newtons.
    monkeypatch.setenv("OPENRAL_SIM_CONTACT_FORCE_CALIBRATION_REF", "fr3-fts-2026-09-04")
    scale, calibrated, reference = contact_force_calibration()
    assert (scale, calibrated, reference) == (2.0, True, "fr3-fts-2026-09-04")

    calibrated_witness = probe_contact_force(
        model,
        data,
        payload_geoms=payload,
        target_root_body=shelf_root,
        target_name="cabinet",
        stamp_ns=1,
    )
    assert calibrated_witness is not None
    assert calibrated_witness.magnitude_calibrated is True
    assert calibrated_witness.calibration_ref == "fr3-fts-2026-09-04"
    assert calibrated_witness.magnitude_n == pytest.approx(2.0 * stock.magnitude_n, rel=1e-9)


@pytest.mark.parametrize("raw_scale", ["not-a-number", "0", "-3.0", "nan", "inf"])
def test_an_unusable_scale_leaves_the_gate_disarmed(
    raw_scale: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail toward the shipped geometry-only behaviour, never toward a guess."""
    monkeypatch.setenv("OPENRAL_SIM_CONTACT_FORCE_N_PER_UNIT", raw_scale)
    monkeypatch.setenv("OPENRAL_SIM_CONTACT_FORCE_CALIBRATION_REF", "named-anyway")
    scale, calibrated, reference = contact_force_calibration()
    assert calibrated is False
    assert reference is None
    assert scale == 1.0


def test_a_contact_on_anything_but_the_declared_target_attests_nothing() -> None:
    """Scoping: force bounds the surface the declaration named, and nothing else.

    The cup is resting on the shelf, in real measurable contact — but asked
    about the sideboard across the room, the producer attests nothing rather
    than reporting the contact it does have under the wrong identity.
    """
    model, data = _settled()
    payload, _, sideboard = _ids(model)

    assert (
        probe_contact_force(
            model,
            data,
            payload_geoms=payload,
            target_root_body=sideboard,
            target_name="sideboard",
            stamp_ns=1,
        )
        is None
    )


def test_a_payload_in_free_flight_attests_nothing() -> None:
    """No contact, no witness — and the gate therefore never arms."""
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # spawned 0.04 m clear of the shelf
    payload, shelf_root, _ = _ids(model)

    assert (
        probe_contact_force(
            model,
            data,
            payload_geoms=payload,
            target_root_body=shelf_root,
            target_name="cabinet",
            stamp_ns=1,
        )
        is None
    )

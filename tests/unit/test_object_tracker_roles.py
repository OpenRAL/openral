"""pro#51: a target and a destination never take each other's box; the older wins."""

from openral_core import ObjectRole, TrackStatus

from tests.unit.test_object_tracker_cover import _CAN, _confirmed_tracker


def _pair():
    tr, _ = _confirmed_tracker(_CAN)
    tgt = tr.store.all()[0]
    assert tgt.role is ObjectRole.TARGET
    fields = {
        k: getattr(tgt, k)
        for k in type(tgt).model_fields
        if k not in ("object_id", "epoch", "status")
    }
    fields["role"] = ObjectRole.DESTINATION
    dst = tr.store.mint(**fields)
    tr.store.replace(dst.model_copy(update={"status": TrackStatus.CONFIRMED}))
    return tr, tgt, dst


def test_a_destination_the_size_of_the_older_target_is_the_target():
    tr, tgt, dst = _pair()
    assert tr._inside_target(dst.box) == tgt.object_id  # a new "destination" mint on it
    assert tr._inside_target(dst.box, dst.object_id) == tgt.object_id  # its refits too


def test_the_older_destination_keeps_its_box_against_a_newer_target():
    tr, tgt, dst = _pair()
    # dst is newer than tgt: the target's refits onto it are not blocked by it
    assert tr._covered_destination(tgt.box, 0, tgt.object_id) is None
    # but a brand-new target mint over it is
    assert tr._covered_destination(tgt.box, 0) == dst.object_id


def test_a_much_smaller_copy_of_a_live_destination_is_refused():
    tr, _, dst = _pair()
    small = dst.box.model_copy(
        update={
            "half_extents": (
                0.3 * dst.box.half_extents[0],
                dst.box.half_extents[1],
                dst.box.half_extents[2],
            )
        }
    )
    assert tr._smaller_copy(dst.label, small) == dst.object_id
    assert tr._smaller_copy(dst.label, dst.box) is None  # same size: a second slot, not a copy

"""A robot joining a shared Isaac sidecar carries only its own totes (``_shared_sidecar_setup``)."""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSConfigError


def test_a_shared_scene_attachment_names_its_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_core.scene_paths import SCENE_ROBOT_ENV
    from openral_sim.backends.isaac_sim import (
        SHARED_SIDECAR_PORT_ENV,
        IsaacSimOptions,
        _shared_sidecar_setup,
    )

    monkeypatch.setenv(SCENE_ROBOT_ENV, "north")
    monkeypatch.setenv(SHARED_SIDECAR_PORT_ENV, "5999")
    tote = {"prim": "Tote", "link": "lift_carriage_link"}
    opts = IsaacSimOptions(
        link_attachments=[{**tote, "robot": "south"}, {**tote, "prim": "Tote_01", "robot": "north"}]
    )
    _robot, _port, mine = _shared_sidecar_setup(opts)
    assert [a.prim for a in mine.link_attachments] == ["Tote_01"]
    with pytest.raises(ROSConfigError, match=r"\['Tote'\] name no robot"):
        _shared_sidecar_setup(IsaacSimOptions(link_attachments=[tote]))

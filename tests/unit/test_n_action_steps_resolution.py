"""One chunk-slicing rule for every policy adapter.

``openral_rskill._vla_core.resolve_n_action_steps`` is the single precedence
every adapter's chunk-slicing line calls: an explicit ``extra["n_action_steps"]``
(the ``--n-action-steps`` CLI override) > ``manifest.n_action_steps`` > the
deprecated ``extra["replan_steps"]`` alias (warns) > the adapter default, then
bounded to ``[1, chunk_size]``. Before it, xr1 / lingbot / rldx / molmoact2 let a
``policy_extras`` key beat the manifest, xvla forced the full chunk and gr00t
read a private queue length.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RSkillManifest
from openral_rskill._vla_core import resolve_n_action_steps
from pydantic import ValidationError

_XR1 = ("xr1-robocasa", "xr1-robocasa365", "xr1-vlabench")


def _manifest(name: str) -> RSkillManifest:
    return RSkillManifest.from_yaml(f"rskills/{name}/rskill.yaml")


def test_manifest_beats_the_adapter_default() -> None:
    m = _manifest("xr1-vlabench")  # n_action_steps: 5, chunk_size: 10
    assert resolve_n_action_steps(m, {}, default=10) == 5


def test_cli_override_beats_the_manifest() -> None:
    m = _manifest("xr1-vlabench")
    assert resolve_n_action_steps(m, {"n_action_steps": 3}, default=10) == 3


def test_replan_steps_is_a_deprecated_alias_below_the_manifest() -> None:
    m = _manifest("xr1-vlabench")
    assert resolve_n_action_steps(m, {"replan_steps": 7}, default=10) == 5
    no_manifest_steps = m.model_copy(update={"n_action_steps": None})
    assert resolve_n_action_steps(no_manifest_steps, {"replan_steps": 7}, default=10) == 7


def test_default_and_clamp() -> None:
    assert resolve_n_action_steps(None, {}, default=8) == 8
    assert resolve_n_action_steps(None, {"n_action_steps": 99}, default=8, chunk_size=16) == 16
    m = _manifest("xr1-robocasa")  # chunk_size 10
    assert resolve_n_action_steps(m, {"n_action_steps": 99}, default=1) == 10
    with pytest.raises(ValueError, match="n_action_steps"):
        resolve_n_action_steps(None, {"n_action_steps": 0}, default=8)


@pytest.mark.parametrize("name", _XR1)
def test_xr1_manifests_no_longer_carry_replan_steps(name: str) -> None:
    m = _manifest(name)
    assert "replan_steps" not in m.policy_extras
    assert m.n_action_steps is not None


def test_manifest_rejects_n_action_steps_above_chunk_size() -> None:
    raw = _manifest("xr1-vlabench").model_dump(mode="json")
    raw["n_action_steps"] = raw["chunk_size"] + 1
    with pytest.raises(ValidationError, match="n_action_steps"):
        RSkillManifest.model_validate(raw)


@pytest.mark.parametrize(
    "path", sorted(Path("rskills").glob("*/rskill.yaml")), ids=lambda p: p.parent.name
)
def test_no_manifest_uses_replan_steps(path: Path) -> None:
    assert "replan_steps" not in RSkillManifest.from_yaml(str(path)).policy_extras

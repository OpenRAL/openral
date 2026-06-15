"""ADR-0051 — detector invocation mode (continuous background vs on-demand locator).

Drives the **real** in-tree detector manifests through the **real** palette
builder + tool-schema renderer (no mocks, CLAUDE.md §1.11):

* the schema default + the four detector manifests' declared modes,
* `build_tool_palette` collecting `mode: continuous` detectors into
  `continuous_detectors` (and never as ExecuteSkill `skills`), while the
  `on_demand` locator is excluded,
* the `locate_in_view` tool description becoming coverage-aware so the LLM is
  told which classes are already tracked in world state.
"""

from __future__ import annotations

from pathlib import Path

from openral_core import RobotCapabilities, RSkillManifest
from openral_core.schemas import DetectorMode
from openral_reasoner.palette import (
    ContinuousDetectorEntry,
    OnDemandDetectorEntry,
    ToolPalette,
    build_tool_palette,
    detector_alias,
    detector_service_segment,
    locate_in_view_service,
)
from openral_reasoner.tool_use import _tool_palette_to_anthropic_tools

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RSKILLS_DIR = _REPO_ROOT / "rskills"


def _load(rskill_id: str) -> RSkillManifest:
    return RSkillManifest.from_yaml(str(_RSKILLS_DIR / rskill_id / "rskill.yaml"))


def _load_intree() -> list[RSkillManifest]:
    return [RSkillManifest.from_yaml(str(p)) for p in sorted(_RSKILLS_DIR.glob("*/rskill.yaml"))]


# ── schema / manifest declarations ──────────────────────────────────────────


def test_detector_mode_defaults_to_continuous() -> None:
    from openral_core.schemas import DetectorContract

    assert DetectorContract(labels=["mug"]).mode is DetectorMode.CONTINUOUS


def test_intree_detectors_declare_expected_modes() -> None:
    expected = {
        "rtdetr-coco-r18": DetectorMode.CONTINUOUS,
        "rtdetr-v2-r50vd": DetectorMode.CONTINUOUS,
        "omdet-turbo-indoor": DetectorMode.CONTINUOUS,
        "omdet-turbo-locator": DetectorMode.ON_DEMAND,
        "locateanything-3b-nf4": DetectorMode.ON_DEMAND,
    }
    for rskill_id, mode in expected.items():
        m = _load(rskill_id)
        assert m.detector is not None and m.detector.mode is mode, rskill_id


# ── palette: continuous detectors surfaced, never as ExecuteSkill tools ──────


def test_palette_collects_continuous_detectors_not_the_on_demand_locator() -> None:
    # A detector is camera-only (all embodiments), so any robot tag surfaces them.
    caps = RobotCapabilities(embodiment_tags=["so100_follower"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)

    cont_ids = {d.rskill_id for d in palette.continuous_detectors}
    assert "OpenRAL/rskill-rtdetr-coco-r18" in cont_ids
    assert "OpenRAL/rskill-omdet-turbo-indoor" in cont_ids
    # The on-demand open-vocab locators are NOT continuous background producers.
    assert "OpenRAL/rskill-locateanything-3b-nf4" not in cont_ids
    assert "OpenRAL/rskill-omdet-turbo-locator" not in cont_ids

    # No detector — continuous or on-demand — is an ExecuteSkill-dispatchable tool.
    skill_ids = {s.rskill_id for s in palette.skills}
    assert not (skill_ids & {d.rskill_id for d in palette.continuous_detectors})
    assert "OpenRAL/rskill-locateanything-3b-nf4" not in skill_ids

    omdet = next(d for d in palette.continuous_detectors if "omdet" in d.rskill_id)
    assert omdet.num_labels > 200  # the curated indoor vocabulary


# ── ADR-0056: on-demand locators surfaced as selectable locate_in_view options ──


def test_palette_surfaces_on_demand_locators_with_aliases() -> None:
    caps = RobotCapabilities(embodiment_tags=["so100_follower"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)

    aliases = {d.alias for d in palette.on_demand_detectors}
    assert "omdet-turbo-locator" in aliases
    assert "locateanything-3b-nf4" in aliases
    # On-demand locators are not continuous producers and not ExecuteSkill tools.
    cont_ids = {d.rskill_id for d in palette.continuous_detectors}
    skill_ids = {s.rskill_id for s in palette.skills}
    for d in palette.on_demand_detectors:
        assert d.rskill_id not in cont_ids
        assert d.rskill_id not in skill_ids
        assert d.alias == detector_alias(d.rskill_id)


def test_detector_alias_and_service_routing() -> None:
    assert detector_alias("OpenRAL/rskill-omdet-turbo-locator") == "omdet-turbo-locator"
    assert detector_alias("plain-name") == "plain-name"
    assert detector_service_segment("omdet-turbo-locator") == "omdet_turbo_locator"
    # Explicit selector → namespaced service.
    assert (
        locate_in_view_service("omdet-turbo-locator")
        == "/openral/perception/omdet_turbo_locator/locate_in_view"
    )
    # Empty selector + a default → the default's namespaced service.
    assert (
        locate_in_view_service("", default="locateanything-3b-nf4")
        == "/openral/perception/locateanything_3b_nf4/locate_in_view"
    )
    # Empty selector + no default → legacy single-detector service (back-compat).
    assert locate_in_view_service("") == "/openral/perception/locate_in_view"


def test_locate_in_view_description_lists_locator_options() -> None:
    palette = ToolPalette(
        detector_available=True,
        on_demand_detectors=(
            OnDemandDetectorEntry(
                rskill_id="OpenRAL/rskill-omdet-turbo-locator",
                alias="omdet-turbo-locator",
                description="fast real-time open-vocab locator",
            ),
            OnDemandDetectorEntry(
                rskill_id="OpenRAL/rskill-locateanything-3b-nf4",
                alias="locateanything-3b-nf4",
                description="high-quality grounding VLM",
            ),
        ),
    )
    desc = {t["name"]: t for t in _tool_palette_to_anthropic_tools(palette)}["locate_in_view"][
        "description"
    ]
    assert "'detector'" in desc
    assert "omdet-turbo-locator" in desc and "locateanything-3b-nf4" in desc


def test_locate_in_view_tool_schema_exposes_detector_field() -> None:
    from openral_core import LocateInViewTool

    schema = LocateInViewTool.model_json_schema()
    assert "detector" in schema["properties"]
    assert LocateInViewTool(query="mug").detector == ""
    assert LocateInViewTool(query="mug", detector="omdet-turbo-locator").detector == (
        "omdet-turbo-locator"
    )


# ── tool_use: locate_in_view description is coverage-aware ───────────────────


def test_locate_in_view_description_lists_continuous_coverage() -> None:
    palette = ToolPalette(
        detector_available=True,
        continuous_detectors=(
            ContinuousDetectorEntry(
                rskill_id="OpenRAL/rskill-omdet-turbo-indoor",
                description="indoor detector",
                objects=("kitchenware", "furniture"),
                num_labels=266,
            ),
        ),
    )
    tools = {t["name"]: t for t in _tool_palette_to_anthropic_tools(palette)}
    desc = tools["locate_in_view"]["description"]
    assert "already tracked continuously" in desc.lower()
    assert "omdet-turbo-indoor" in desc
    assert "266 classes" in desc and "kitchenware" in desc


def test_locate_in_view_description_plain_without_continuous_detectors() -> None:
    # detector_available but no continuous bank → no coverage clause appended.
    palette = ToolPalette(detector_available=True)
    tools = {t["name"]: t for t in _tool_palette_to_anthropic_tools(palette)}
    assert "already tracked continuously" not in tools["locate_in_view"]["description"].lower()


# ── node wiring policy (ADR-0051): continuous publishes, on_demand serves ────


def test_detector_node_wiring_continuous_publishes_not_serves() -> None:
    from openral_runner.backends.gstreamer.detector_factory import detector_node_wiring

    w = detector_node_wiring(DetectorMode.CONTINUOUS)
    assert w.run_continuous_leg is True
    assert w.serve_on_demand is False


def test_detector_node_wiring_on_demand_serves_not_publishes() -> None:
    from openral_runner.backends.gstreamer.detector_factory import detector_node_wiring

    w = detector_node_wiring(DetectorMode.ON_DEMAND)
    assert w.run_continuous_leg is False
    assert w.serve_on_demand is True


def test_node_wiring_matches_intree_detector_manifests() -> None:
    # End-to-end: each in-tree detector's declared mode maps to the expected
    # node wiring (continuous bank publishes; locators serve locate_in_view).
    from openral_runner.backends.gstreamer.detector_factory import detector_node_wiring

    publishes = {"rtdetr-coco-r18", "rtdetr-v2-r50vd", "omdet-turbo-indoor"}
    serves = {"omdet-turbo-locator", "locateanything-3b-nf4"}
    for rid in publishes | serves:
        m = _load(rid)
        w = detector_node_wiring(m.detector.mode)
        assert w.run_continuous_leg is (rid in publishes), rid
        assert w.serve_on_demand is (rid in serves), rid

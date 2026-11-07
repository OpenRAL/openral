"""``ManifestSkillResolver.resolve`` reads the manifest it is asked for, fresh, by name.

Real in-tree manifests copied into per-test search paths; no Hub access (an in-tree
hit never falls through, and ``HF_HUB_OFFLINE=1`` makes any accidental Hub call fail).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError

pytest.importorskip("rclpy")

_REPO = Path(__file__).resolve().parents[2]
_VLA = _REPO / "rskills" / "smolvla-libero" / "rskill.yaml"
_NAME = "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"


def _copy(root: Path, text: str, *, skill_dir: str = "skill") -> Path:
    path = root / skill_dir / "rskill.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def test_an_edited_manifest_is_gated_as_edited(tmp_path: Path) -> None:
    """The gate sees the file as it is now, not as it was at the first dispatch."""
    from openral_rskill_ros.rskill_runner_node import ManifestSkillResolver, check_manifest_gates

    source = _VLA.read_text(encoding="utf-8")
    path = _copy(tmp_path, source)
    resolver = ManifestSkillResolver(search_paths=[tmp_path], hub_fallback=False)
    assert resolver.resolve(rskill_id=_NAME, revision="").manifest.license.value == "apache-2.0"

    path.write_text(
        source.replace('license: "apache-2.0"', 'license: "nvidia_non_commercial"'), "utf-8"
    )
    resolved = resolver.resolve(rskill_id=_NAME, revision="")
    assert resolved.manifest.license.value == "nvidia_non_commercial"
    franka = RobotDescription.from_yaml(str(_REPO / "robots/franka_panda/robot.yaml"))
    with pytest.raises(ROSConfigError, match=r"(?i)commercial"):
        check_manifest_gates(resolved.manifest, description=franka, commercial_deployment=True)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        # A schema-0.2 role mismatch: refused, with the field-level reason.
        (
            lambda t: re.sub(r"(?m)^role:.*$", 'role: "s2"', t).replace(
                'schema_version: "0.1"', 'schema_version: "0.2"'
            ),
            "kind='vla' requires role='s1'",
        ),
        # Broken YAML after the name line.
        (lambda t: t + "\nbroken: [\n", "invalid YAML"),
    ],
)
def test_an_invalid_intree_manifest_is_reported_not_replaced_by_the_hub(
    tmp_path: Path, edit: object, match: str
) -> None:
    """Found by name, an invalid in-tree manifest raises its own error — no Hub fallback."""
    from openral_rskill_ros.rskill_runner_node import ManifestSkillResolver

    _copy(tmp_path, edit(_VLA.read_text(encoding="utf-8")))  # type: ignore[operator]  # reason: parametrized callable
    resolver = ManifestSkillResolver(search_paths=[tmp_path], hub_fallback=True)
    with pytest.raises(ROSConfigError, match=re.escape(match)):
        resolver.resolve(rskill_id=_NAME, revision="")


def test_the_first_search_path_wins(tmp_path: Path) -> None:
    """Same name in two search paths: the earlier one, as the reasoner's palette seed has it."""
    from openral_rskill_ros.rskill_runner_node import ManifestSkillResolver

    source = _VLA.read_text(encoding="utf-8")
    first = _copy(tmp_path / "site", source)
    _copy(tmp_path / "stock", source)
    resolver = ManifestSkillResolver(
        search_paths=[tmp_path / "site", tmp_path / "stock"], hub_fallback=False
    )
    assert resolver.resolve(rskill_id=_NAME, revision="").manifest_path == first


def test_an_unknown_id_names_the_known_ones(tmp_path: Path) -> None:
    from openral_rskill_ros.rskill_runner_node import ManifestSkillResolver

    _copy(tmp_path, _VLA.read_text(encoding="utf-8"))
    resolver = ManifestSkillResolver(search_paths=[tmp_path], hub_fallback=False)
    with pytest.raises(ROSConfigError, match=re.escape(_NAME)):
        resolver.resolve(rskill_id="OpenRAL/rskill-not-here", revision="")

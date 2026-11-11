"""Config-named code (``hal.sim`` / ``hal.real`` / ``composition.composer``) is openral_* only.

A manifest or scene names a ``"module:attribute"`` that OpenRAL imports and calls with
config-supplied kwargs; ``os:system`` there was arbitrary code execution. Each test
loads a real shipped fixture, swaps in a hostile reference and expects a refusal at
schema load AND at the import seam (``build_hal`` / the HAL node's composer call).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core import DeployScene, RobotDescription, SceneComposition
from openral_core.code_refs import ALLOW_EXTERNAL_PLUGINS_ENV, check_code_ref, import_code_ref
from openral_core.exceptions import ROSConfigError

REPO = Path(__file__).resolve().parents[2]
_UR5E = REPO / "robots" / "ur5e" / "robot.yaml"
_OPENARM_SCENE = REPO / "scenes" / "deploy" / "openarm_tabletop.yaml"

_HOSTILE = ["os:system", "subprocess:Popen", "builtins:exec", "pkg.mod:compose"]


@pytest.mark.parametrize("ref", _HOSTILE)
def test_a_reference_outside_openral_is_refused(ref: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    with pytest.raises(ROSConfigError, match=r"hal\.real .*outside OpenRAL"):
        check_code_ref(ref, setting="hal.real")
    with pytest.raises(ROSConfigError, match="outside OpenRAL"):
        import_code_ref(ref, setting="hal.real")


@pytest.mark.parametrize("ref", ["openral_hal", "openral_hal:", "openral_hal:x;y", "a b:c"])
def test_a_malformed_reference_is_refused(ref: str) -> None:
    with pytest.raises(ROSConfigError, match="malformed"):
        check_code_ref(ref, setting="composition.composer")


def test_openral_and_pro_packages_are_admitted() -> None:
    assert check_code_ref("openral_hal.ur_real:UR5eRealHAL", setting="hal.real")
    # OpenRAL Pro's packages share the prefix; no Pro install is needed to check it.
    assert check_code_ref("openral_pro_trt.hal:X", setting="hal.real")
    # A look-alike top-level module is not an OpenRAL package.
    with pytest.raises(ROSConfigError):
        check_code_ref("openralx.hal:X", setting="hal.real")


def test_the_opt_in_admits_an_external_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_EXTERNAL_PLUGINS_ENV, "1")
    assert import_code_ref("json:dumps", setting="hal.sim") is __import__("json").dumps


def _robot_with(hal: dict[str, object]) -> dict[str, object]:
    raw: dict[str, object] = yaml.safe_load(_UR5E.read_text())
    raw["hal"] = {**raw["hal"], **hal}  # type: ignore[dict-item]  # reason: YAML mapping
    return raw


@pytest.mark.parametrize("field", ["sim", "real"])
def test_a_robot_manifest_naming_foreign_code_does_not_load(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    with pytest.raises(ROSConfigError, match=rf"hal\.{field} = 'os:system'"):
        RobotDescription.model_validate(_robot_with({field: "os:system"}))


def test_build_hal_rechecks_a_reference_that_skipped_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openral_hal import build_hal

    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    description = RobotDescription.from_yaml(str(_UR5E))
    # model_construct-style mutation bypasses the validator; the import seam must not.
    object.__setattr__(description.hal, "real", "os:system")
    with pytest.raises(ROSConfigError, match=r"hal\.real = 'os:system'"):
        build_hal(description, mode="real")


def test_a_scene_composer_naming_foreign_code_does_not_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    raw = yaml.safe_load(_OPENARM_SCENE.read_text())
    assert raw["composition"]["composer"].startswith("openral_sim.")
    DeployScene.model_validate(raw)  # the shipped scene loads
    raw["composition"]["composer"] = "os:system"
    with pytest.raises(ROSConfigError, match=r"composition\.composer = 'os:system'"):
        DeployScene.model_validate(raw)
    # The HAL node's wire format (scene_composition_json) is validated the same way.
    with pytest.raises(ROSConfigError, match="outside OpenRAL"):
        SceneComposition.model_validate_json('{"composer": "os:system", "params": {}}')

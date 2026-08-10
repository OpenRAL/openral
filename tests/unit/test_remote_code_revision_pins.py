"""Custom-code rSkills must execute an immutable checkpoint revision."""

from pathlib import Path

from openral_core import RSkillManifest

_ROOT = Path(__file__).parents[2]
_CUSTOM_CODE_FAMILIES = {"openvla", "xr1"}


def test_custom_code_rskills_pin_weights_revision() -> None:
    checked: list[str] = []
    for manifest_path in sorted((_ROOT / "rskills").glob("*/rskill.yaml")):
        manifest = RSkillManifest.from_yaml(str(manifest_path))
        quantization_extra = (
            manifest.quantization.extra if manifest.quantization is not None else {}
        )
        executes_remote_code = (
            manifest.model_family in _CUSTOM_CODE_FAMILIES
            or quantization_extra.get("trust_remote_code") is True
        )
        if not executes_remote_code:
            continue
        checked.append(manifest.name)
        assert manifest.weights_uri is not None
        assert "@" in manifest.weights_uri, (
            f"{manifest_path}: custom-code weights_uri must pin an immutable revision"
        )
    assert checked

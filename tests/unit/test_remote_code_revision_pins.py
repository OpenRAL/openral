"""Custom-code rSkills must execute an immutable checkpoint revision."""

import re
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


_SHA_PIN = re.compile(r"@[0-9a-f]{40}$")


def test_hf_rskills_pin_weights_revision() -> None:
    """Every ``hf://`` weights_uri pins a commit SHA so a run replays from its trace (§1.8)."""
    unpinned = []
    for manifest_path in sorted((_ROOT / "rskills").glob("*/rskill.yaml")):
        if manifest_path.parent.name == "template":  # placeholder repo id, never loaded
            continue
        weights_uri = RSkillManifest.from_yaml(str(manifest_path)).weights_uri
        if weights_uri and weights_uri.startswith("hf://") and not _SHA_PIN.search(weights_uri):
            unpinned.append(manifest_path.parent.name)
    assert not unpinned, f"hf:// weights_uri must pin '@<40-hex sha>': {unpinned}"


# These manifests declare policy_{pre,post}processor.json files their repos do
# not contain at any revision checked (2026-11-07), so there is nothing to pin.
# Fix the manifest (drop the block or point it at a real file), then delete it here.
_PROCESSOR_FILES_ABSENT_UPSTREAM = frozenset(
    {
        "diffusion-pusht",
        "molmoact2-libero-nf4",
        "molmoact2-so101-nf4",
        "rldx1-ft-gr1-nf4",
        "rldx1-ft-libero-nf4",
        "rldx1-ft-rc365-nf4",
        "rldx1-ft-simpler-widowx-nf4",
    }
)
_PROCESSOR_PIN = re.compile(r"^hf://[^/]+/[^/@]+@[0-9a-f]{40}/")


def test_hf_rskill_processor_files_pin_revision() -> None:
    """Processor JSONs load at a pinned SHA too, or the run is only half replayable."""
    unpinned = set()
    for manifest_path in sorted((_ROOT / "rskills").glob("*/rskill.yaml")):
        if manifest_path.parent.name == "template":
            continue
        processors = RSkillManifest.from_yaml(str(manifest_path)).processors
        if processors is None:
            continue
        for uri in (processors.preprocessor_uri, processors.postprocessor_uri):
            if not _PROCESSOR_PIN.match(uri):
                unpinned.add(manifest_path.parent.name)
    assert unpinned == _PROCESSOR_FILES_ABSENT_UPSTREAM, (
        f"unpin: {sorted(unpinned - _PROCESSOR_FILES_ABSENT_UPSTREAM)}; "
        f"now pinned, drop from the list: {sorted(_PROCESSOR_FILES_ABSENT_UPSTREAM - unpinned)}"
    )

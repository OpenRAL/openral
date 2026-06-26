"""Schema export tool — generates JSON Schema files for every public openral_core model.

Run with:
    uv run python tools/schema_export.py

Writes to docs/reference/schemas/<ModelName>.json and docs/reference/schemas/all.json.
CI compares the output of this script against committed files; any drift fails the build.

Schema versioning rules (pre-1.0):
  - Breaking change (field removed, type narrowed, required added) → bump MINOR.
  - Additive change (optional field added, enum value added) → bump PATCH.
  - Both rules flip to MAJOR/MINOR at v1.0.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# ── Ensure the workspace packages are importable when run from repo root ──────
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "core" / "src"))

import openral_core  # reason: path manipulation above
from openral_core.schemas import (
    Action,
    ActionRepresentation,
    ActionSpec,
    BenchmarkMetadata,
    BenchmarkScene,
    CameraSimPlacement,
    ControlMode,
    DeadlineOverrunPolicy,
    DeployScene,
    DetectedObject,
    EmbodimentKind,
    EndEffectorSpec,
    ExecuteRskillTool,
    FrameEncoding,
    HalConfig,
    Hand,
    IntrinsicsPinhole,
    JointSpec,
    JointState,
    JointType,
    ObservationSpec,
    PhysicsBackend,
    Pose6D,
    ProtocolSpec,
    RewardContract,
    RoboCasaBackendOptions,
    RobotCapabilities,
    RobotDescription,
    RobotEnvironment,
    RSkillEvalBenchmark,
    RSkillEvalResult,
    RSkillEvalSource,
    RSkillManifest,
    RunResult,
    SafetyEnvelope,
    SceneSpec,
    SensorBundle,
    SensorFrame,
    SensorModality,
    SensorReaderBackend,
    SensorReaderConfig,
    SensorSpec,
    SimEnvironment,
    SimScene,
    StateRepresentation,
    TaskSpec,
    TickResult,
    VLASpec,
    WorldState,
)

# ── Models to export (ordered: enums first, then leaf → root) ────────────────
_ENUM_TYPES: list[type] = [
    EmbodimentKind,
    JointType,
    ControlMode,
    SensorModality,
    Hand,
    StateRepresentation,
    ActionRepresentation,
    PhysicsBackend,
    FrameEncoding,
    SensorReaderBackend,
    DeadlineOverrunPolicy,
]

_MODEL_TYPES: list[type] = [
    IntrinsicsPinhole,
    CameraSimPlacement,
    SensorSpec,
    SensorBundle,
    JointSpec,
    EndEffectorSpec,
    RobotCapabilities,
    SafetyEnvelope,
    ObservationSpec,
    ActionSpec,
    RobotDescription,
    RewardContract,
    ExecuteRskillTool,
    RSkillManifest,
    JointState,
    Pose6D,
    DetectedObject,
    SensorFrame,
    WorldState,
    Action,
    SceneSpec,
    RoboCasaBackendOptions,
    TaskSpec,
    VLASpec,
    SimEnvironment,
    DeployScene,
    SimScene,
    BenchmarkMetadata,
    BenchmarkScene,
    ProtocolSpec,
    RSkillEvalSource,
    RSkillEvalBenchmark,
    RSkillEvalResult,
    SensorReaderConfig,
    HalConfig,
    RobotEnvironment,
    TickResult,
    RunResult,
]

_OUT_DIR = Path(__file__).parent.parent / "docs" / "reference" / "schemas"


def _enum_schema(cls: type) -> dict[str, Any]:
    """Build a minimal JSON Schema for a str Enum."""
    members = [m.value for m in cls]  # type: ignore[attr-defined]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": cls.__name__,
        "description": (cls.__doc__ or "").strip().split("\n")[0],
        "type": "string",
        "enum": members,
    }


def export_schemas(out_dir: Path = _OUT_DIR) -> dict[str, Any]:
    """Export JSON Schema for every public model.

    Args:
        out_dir: Directory to write schema files into.

    Returns:
        Mapping of model name → schema dict (also written to disk).

    Raises:
        RuntimeError: If schema generation fails for any model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_schemas: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "openral_core",
        "version": openral_core.__version__,
        "schemas": {},
    }

    # Enums
    for cls in _ENUM_TYPES:
        schema = _enum_schema(cls)
        out_path = out_dir / f"{cls.__name__}.json"
        out_path.write_text(json.dumps(schema, indent=2) + "\n")
        print(f"  wrote {out_path.relative_to(out_dir.parent.parent.parent)}")
        all_schemas["schemas"][cls.__name__] = schema

    # Pydantic models
    for cls in _MODEL_TYPES:
        try:
            schema = cls.model_json_schema()  # type: ignore[attr-defined]  # reason: list[type] is unparameterised; model_json_schema exists on all entries
        except Exception as exc:
            raise RuntimeError(f"Failed to generate schema for {cls.__name__}") from exc

        # Inject $schema and version metadata
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema.setdefault("description", (cls.__doc__ or "").strip().split("\n")[0])

        out_path = out_dir / f"{cls.__name__}.json"
        out_path.write_text(json.dumps(schema, indent=2) + "\n")
        print(f"  wrote {out_path.relative_to(out_dir.parent.parent.parent)}")
        all_schemas["schemas"][cls.__name__] = schema

    # Combined
    combined_path = out_dir / "all.json"
    combined_path.write_text(json.dumps(all_schemas, indent=2) + "\n")
    print(f"  wrote {combined_path.relative_to(out_dir.parent.parent.parent)}")

    return all_schemas


def check_drift(out_dir: Path = _OUT_DIR) -> bool:
    """Return True if the on-disk schemas match what would be generated now.

    Used by CI: exits non-zero if schemas are stale.

    Args:
        out_dir: Directory containing committed schema files.

    Returns:
        True if no drift detected, False otherwise.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        export_schemas(out_dir=tmp_dir)
        drift = False
        for generated in tmp_dir.iterdir():
            committed = out_dir / generated.name
            if not committed.exists():
                print(f"  DRIFT: {generated.name} is new (not committed)", file=sys.stderr)
                drift = True
            elif committed.read_text() != generated.read_text():
                print(f"  DRIFT: {generated.name} differs from committed", file=sys.stderr)
                drift = True
        for committed in out_dir.iterdir():
            if not (tmp_dir / committed.name).exists():
                msg = f"  DRIFT: {committed.name} is committed but no longer generated"
                print(msg, file=sys.stderr)
                drift = True
    return not drift


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export openral_core JSON Schemas.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for drift between generated and committed schemas (CI mode).",
    )
    args = parser.parse_args()

    if args.check:
        print("Checking schema drift...")
        ok = check_drift()
        if ok:
            print("No drift detected.")
            sys.exit(0)
        else:
            print("Schema drift detected — run `just schema-export` and commit.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Exporting schemas to {_OUT_DIR.relative_to(Path.cwd())} ...")
        export_schemas()
        print("Done.")

"""Import authored object metadata from external USD without starting Isaac Sim."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from urllib.parse import unquote, urlsplit

from openral_core.exceptions import ROSConfigError
from openral_core.schemas import DeployScene, USDObjectCatalog

__all__ = ["USD_OBJECT_CATALOG_CUSTOM_DATA_KEY", "load_usd_object_catalog"]

USD_OBJECT_CATALOG_CUSTOM_DATA_KEY = "openral:object_catalog_json"
"""CustomData key containing a JSON-encoded :class:`USDObjectCatalog`."""

# Keep the record below a conservative process-argument budget: deploy passes
# this validated JSON through the ROS launch argument to the tracker.
_MAX_CATALOG_BYTES = 64_000
_MAX_READER_RESULT_BYTES = 256_000
_READER_TIMEOUT_S = 30.0
_USD_READER_HELPER = Path(__file__).with_name("_usd_catalog_reader.py")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _read_with_external_python(  # noqa: PLR0912  # reason: subprocess boundary validation
    path: Path, *, catalog_prim_path: str, reader_python: str
) -> dict[str, object]:
    executable = Path(reader_python)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ROSConfigError(
            f"object_catalog.reader_python must name an executable file: {reader_python!r}"
        )
    with tempfile.TemporaryDirectory(prefix="openral-usd-catalog-") as directory:
        request_path = Path(directory) / "request.json"
        result_path = Path(directory) / "result.json"
        request_path.write_text(
            json.dumps(
                {
                    "usd_path": str(path),
                    "catalog_prim_path": catalog_prim_path,
                    "max_catalog_bytes": _MAX_CATALOG_BYTES,
                    "max_result_bytes": _MAX_READER_RESULT_BYTES,
                }
            ),
            encoding="utf-8",
        )
        try:
            process = subprocess.Popen(
                [str(executable), str(_USD_READER_HELPER), str(request_path), str(result_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
            returncode = process.wait(timeout=_READER_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise ROSConfigError(
                f"object_catalog.reader_python timed out after {_READER_TIMEOUT_S:g} seconds"
            ) from exc
        except OSError as exc:
            raise ROSConfigError(
                f"could not start object_catalog.reader_python {reader_python!r}: {exc}"
            ) from exc
        if returncode != 0:
            raise ROSConfigError(f"object_catalog.reader_python exited with status {returncode}")
        try:
            if result_path.stat().st_size > _MAX_READER_RESULT_BYTES:
                raise ROSConfigError(
                    f"object_catalog reader result exceeds {_MAX_READER_RESULT_BYTES} bytes"
                )
            response = json.loads(
                result_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
            )
        except ROSConfigError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise ROSConfigError(f"invalid object_catalog reader result: {exc}") from exc
    if not isinstance(response, dict) or any(not isinstance(key, str) for key in response):
        raise ROSConfigError("invalid object_catalog reader result: expected a JSON object")
    error = response.get("error")
    if isinstance(error, str):
        raise ROSConfigError(f"could not read object catalog with reader_python: {error}")
    if set(response) != {"payload", "catalog_prim_loaded", "referenced_prims"}:
        raise ROSConfigError("invalid object_catalog reader result: unexpected fields")
    if (
        not isinstance(response["payload"], str)
        or type(response["catalog_prim_loaded"]) is not bool
        or not isinstance(response["referenced_prims"], list)
    ):
        raise ROSConfigError("invalid object_catalog reader result: wrong field types")
    return response


def _check_external_prim_results(
    response: dict[str, object], references: list[tuple[str, str]], path: Path, prim_path: str
) -> None:
    if not response["catalog_prim_loaded"]:
        raise ROSConfigError(
            f"object_catalog.prim_path={prim_path!r} is missing or unloaded in {path}; "
            "catalog metadata must be on a non-payload prim"
        )
    results = response["referenced_prims"]
    if not isinstance(results, list) or len(results) != len(references):
        raise ROSConfigError("invalid object_catalog reader result: referenced prim count mismatch")
    for (name, ref_path), item in zip(references, results, strict=True):
        if not isinstance(item, dict) or set(item) != {"path", "exists", "loaded"}:
            raise ROSConfigError("invalid object_catalog reader result: malformed prim status")
        if (
            item["path"] != ref_path
            or type(item["exists"]) is not bool
            or type(item["loaded"]) is not bool
        ):
            raise ROSConfigError("invalid object_catalog reader result: prim status mismatch")
        if not item["exists"] or not item["loaded"]:
            raise ROSConfigError(
                f"USD catalog {name!r} references missing or unloaded prim {ref_path!r}"
            )


def _catalog_path(scene: DeployScene, scene_path: Path) -> Path:
    reference = scene.object_catalog
    if reference is None:
        raise ROSConfigError("load_usd_object_catalog requires scene.object_catalog")
    uri = reference.usd or scene.scene.assets_uri
    if not uri:
        raise ROSConfigError(
            "object_catalog.usd is required when scene.assets_uri does not name an external USD"
        )
    parsed = urlsplit(uri)
    if parsed.scheme and parsed.scheme != "file":
        raise ROSConfigError(
            f"object_catalog.usd={uri!r} is not a local/file USD; standalone OpenUSD loading "
            "currently needs a local asset path"
        )
    if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
        raise ROSConfigError(
            f"object_catalog.usd={uri!r} names a remote file host; use a local USD path"
        )
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = scene_path.resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise ROSConfigError(f"object catalog USD does not exist: {path}")
    return path


def load_usd_object_catalog(  # noqa: PLR0912  # reason: fail-fast boundary validation
    scene: DeployScene, *, scene_path: str | Path
) -> USDObjectCatalog | None:
    """Read and validate the authored customData catalog from a DeployScene's USD.

    The optional ``usd-core`` package supplies standalone ``pxr.Usd``. No
    simulator app, live transform, or simulator-only module is imported.

    Args:
        scene: Validated deploy scene; no catalog returns ``None`` without importing USD.
        scene_path: YAML path used to resolve a relative catalog/USD path.

    Raises:
        ROSConfigError: On missing USD support, malformed USD/catalog metadata, a source
            revision mismatch, or catalog prim paths that do not exist in the stage.

    Example:
        >>> from openral_core import DeployScene, PhysicsBackend, SceneSpec
        >>> scene = DeployScene(scene=SceneSpec(id="demo", backend=PhysicsBackend.ISAACSIM))
        >>> load_usd_object_catalog(scene, scene_path="unused.yaml") is None
        True
    """
    reference = scene.object_catalog
    if reference is None:
        return None
    path = _catalog_path(scene, Path(scene_path))
    external_response: dict[str, object] | None = None
    if reference.reader_python is not None:
        external_response = _read_with_external_python(
            path, catalog_prim_path=reference.prim_path, reader_python=reference.reader_python
        )
        payload = external_response["payload"]
    else:
        try:
            from pxr import Usd  # noqa: PLC0415  # reason: optional scene-usd group stays lazy
        except ImportError as exc:
            raise ROSConfigError(
                "loading DeployScene.object_catalog requires standalone OpenUSD; install it with "
                "`just sync --group scene-usd`, or configure object_catalog.reader_python "
                "to use another USD Python (no Isaac Sim app is required)"
            ) from exc
        try:
            stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
            if stage is None:
                raise ROSConfigError(f"could not open object catalog USD: {path}")
            if reference.prim_path == "/":
                # Sdf does not permit customData on its pseudo-root spec. The USD
                # stage-level equivalent is customLayerData on the root layer.
                namespace, key = USD_OBJECT_CATALOG_CUSTOM_DATA_KEY.split(":", maxsplit=1)
                payload = stage.GetRootLayer().customLayerData.get(namespace, {}).get(key)
            else:
                prim = stage.GetPrimAtPath(reference.prim_path)
                if not prim or not prim.IsValid() or not prim.IsLoaded():
                    raise ROSConfigError(
                        f"object_catalog.prim_path={reference.prim_path!r} is missing or unloaded "
                        f"in {path}; catalog metadata must be on a non-payload prim"
                    )
                payload = prim.GetCustomDataByKey(USD_OBJECT_CATALOG_CUSTOM_DATA_KEY)
        except ROSConfigError:
            raise
        except Exception as exc:
            raise ROSConfigError(f"could not read object catalog from USD {path}: {exc}") from exc
    if not isinstance(payload, str):
        raise ROSConfigError(
            f"USD prim {reference.prim_path!r} must carry string customData "
            f"{USD_OBJECT_CATALOG_CUSTOM_DATA_KEY!r}"
        )
    if len(payload.encode("utf-8")) > _MAX_CATALOG_BYTES:
        raise ROSConfigError(f"USD object catalog exceeds {_MAX_CATALOG_BYTES} bytes")
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        catalog = USDObjectCatalog.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise ROSConfigError(f"invalid USD object catalog in {path}: {exc}") from exc
    if catalog.source_revision != reference.expected_revision:
        raise ROSConfigError(
            f"USD object catalog revision {catalog.source_revision!r} does not match "
            f"object_catalog.expected_revision={reference.expected_revision!r}"
        )
    references = [(entry.name, entry.prim_path) for entry in catalog.objects] + [
        (entry.name, entry.prim_path) for entry in catalog.workspaces
    ]
    if external_response is not None:
        _check_external_prim_results(external_response, references, path, reference.prim_path)
        return catalog
    for name, prim_path in references:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not prim.IsLoaded():
            raise ROSConfigError(
                f"USD catalog {name!r} references missing or unloaded prim {prim_path!r}"
            )
    return catalog

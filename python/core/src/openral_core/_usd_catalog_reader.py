"""Small stdlib/OpenUSD worker for hosts whose Python cannot import ``pxr``."""

from __future__ import annotations

import json
import sys


def _write_response(path: str, response: dict[str, object], limit: int) -> None:
    encoded = json.dumps(response, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > limit:
        response = {"error": "reader result exceeds configured byte limit"}
        encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as output:
        output.write(encoded)


def _main() -> int:  # noqa: PLR0912  # reason: this isolated worker validates its tiny file protocol
    request_path, response_path = sys.argv[1:3]
    max_result_bytes = 256_000
    try:
        with open(request_path, encoding="utf-8") as source:
            request = json.load(source)
        if not isinstance(request, dict):
            raise ValueError("reader request must be an object")
        usd_path = request["usd_path"]
        catalog_prim_path = request["catalog_prim_path"]
        max_catalog_bytes = request["max_catalog_bytes"]
        max_result_bytes = request["max_result_bytes"]
        if (
            not isinstance(usd_path, str)
            or not isinstance(catalog_prim_path, str)
            or type(max_catalog_bytes) is not int
            or type(max_result_bytes) is not int
        ):
            raise ValueError("reader request has invalid field types")

        from pxr import Usd  # noqa: PLC0415, I001  # reason: report missing USD support over the worker protocol

        stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadNone)
        if stage is None:
            raise ValueError("could not open USD stage")
        catalog_prim_loaded = True
        if catalog_prim_path == "/":
            namespace, key = "openral", "object_catalog_json"
            payload = stage.GetRootLayer().customLayerData.get(namespace, {}).get(key)
        else:
            prim = stage.GetPrimAtPath(catalog_prim_path)
            catalog_prim_loaded = bool(prim and prim.IsValid() and prim.IsLoaded())
            if not catalog_prim_loaded:
                raise ValueError("catalog prim is missing or unloaded")
            payload = prim.GetCustomDataByKey("openral:object_catalog_json")
        if not isinstance(payload, str):
            raise ValueError("catalog metadata must be a string")
        if len(payload.encode("utf-8")) > max_catalog_bytes:
            raise ValueError("catalog payload exceeds configured byte limit")

        references: list[str] = []
        try:
            raw = json.loads(payload)
        except (TypeError, ValueError):
            raw = None  # Host reports the canonical catalog JSON error.
        if isinstance(raw, dict):
            for collection in ("objects", "workspaces"):
                entries = raw.get(collection)
                if isinstance(entries, list):
                    references.extend(
                        entry["prim_path"]
                        for entry in entries
                        if isinstance(entry, dict) and isinstance(entry.get("prim_path"), str)
                    )
        statuses = []
        for prim_path in references:
            prim = stage.GetPrimAtPath(prim_path)
            statuses.append(
                {
                    "path": prim_path,
                    "exists": bool(prim and prim.IsValid()),
                    "loaded": bool(prim and prim.IsValid() and prim.IsLoaded()),
                }
            )
        _write_response(
            response_path,
            {
                "payload": payload,
                "catalog_prim_loaded": catalog_prim_loaded,
                "referenced_prims": statuses,
            },
            max_result_bytes,
        )
    except Exception as exc:
        _write_response(response_path, {"error": str(exc)[:1_024]}, max_result_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Export an Isaac scene's static colliders as declared static geometry (base-frame boxes).

Boots the scene's Isaac env exactly as ``openral deploy sim`` does
(``openral_hal.sim_bringup.build_sim_env_from_yaml``), asks the sidecar for every static
collider's oriented bounding box in the manifest ``base_frame``
(``IsaacManifestScene.static_collider_boxes``), and prints them, or writes them as YAML
(``--out``) for a deploy scene's declared static geometry
(``docs/reference/manipulation-workspace-contact.md`` §3.1).

Each box is a superset of its collider (the USD bound), the fail-closed direction for
geometry a scene adds to the kernel's world. Needs Isaac Sim (the sidecar); run::

    python tools/isaac_static_geometry.py --config scenes/deploy/local/<scene>.yaml \\
        [--out static.yaml] [--max-distance 1.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def to_declared(box: dict[str, Any]) -> dict[str, Any]:
    """A sidecar ``kind: box`` collider as a declared static-geometry entry (base frame).

    Example:
        >>> e = to_declared(
        ...     {
        ...         "path": "/World/env/Wall",
        ...         "center_m": [1.0, 0.0, 0.2],
        ...         "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ...         "half_extents_m": [0.1, 1.0, 0.5],
        ...         "collision_enabled": True,
        ...     }
        ... )
        >>> e["kind"], e["xyz"], e["quat_xyzw"], e["half_extents"], e["source"]
        ('box', [1.0, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0], [0.1, 1.0, 0.5], '/World/env/Wall')
    """
    from openral_core.geometry import rotation_to_quat_wxyz

    w, x, y, z = rotation_to_quat_wxyz(np.asarray(box["rotation"], dtype=np.float64))
    return {
        "kind": "box",
        "source": box["path"],
        "xyz": [float(v) for v in box["center_m"]],
        "quat_xyzw": [round(float(v), 9) for v in (x, y, z, w)],
        "half_extents": [float(v) for v in box["half_extents_m"]],
    }


def mesh_bounds(vertices: Any) -> tuple[list[float], list[float]]:
    """Centre and half-extents of a vertex set's axis-aligned bound.

    Example:
        >>> mesh_bounds([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        ([1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
    """
    v = np.asarray(vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return [float(x) for x in (lo + hi) / 2.0], [float(x) for x in (hi - lo) / 2.0]


def nearest_distance_m(entry: dict[str, Any]) -> float:
    """Lower bound on the entry's distance from the base-frame origin (centre minus radius).

    Example:
        >>> nearest_distance_m({"xyz": [3.0, 0.0, 0.0], "half_extents": [1.0, 0.0, 0.0]})
        2.0
    """
    return max(
        0.0, float(np.linalg.norm(entry["xyz"])) - float(np.linalg.norm(entry["half_extents"]))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Deploy scene YAML (Isaac backend).")
    parser.add_argument("--out", help="Write the boxes here as YAML; else print a table.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Keep only boxes whose nearest point may lie within this many metres of base_frame.",
    )
    args = parser.parse_args(argv)

    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_core.exceptions import ROSConfigError
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(args.config, robot_id_fallback=None)
    try:
        export = getattr(env, "static_collider_boxes", None)
        if not callable(export):
            raise ROSConfigError(
                f"{args.config}: the scene's backend exports no static colliders (Isaac only)"
            )
        env.reset(seed=seed)
        reply = export()
    finally:
        env.close()
    entries: list[dict[str, Any]] = []
    meshes: dict[str, dict[str, Any]] = {}
    for c in reply["boxes"]:
        if not c.get("collision_enabled", True):
            continue
        if c.get("kind") == "mesh":
            center, half = mesh_bounds(c["vertices_m"])
            name = c["path"].strip("/").replace("/", "__") + ".npz"
            meshes[name] = c
            entries.append(
                {
                    "kind": "mesh",
                    "source": c["path"],
                    "file": name,
                    "xyz": center,
                    "half_extents": half,
                    "triangles": len(c["triangles"]),
                }
            )
        else:
            entries.append(to_declared(c))
    if args.max_distance is not None:
        entries = [e for e in entries if nearest_distance_m(e) <= args.max_distance]
    entries.sort(key=nearest_distance_m)
    if args.out:
        out = Path(args.out)
        for e in entries:
            if e["kind"] == "mesh":
                c = meshes[e["file"]]
                np.savez_compressed(
                    out.parent / e["file"],
                    vertices=np.asarray(c["vertices_m"], dtype=np.float32),
                    triangles=np.asarray(c["triangles"], dtype=np.int32),
                )
                # Only the box-like summary goes in the YAML; the surface is in the file.
                del e["xyz"], e["half_extents"]
        out.write_text(yaml.safe_dump({"static_geometry": entries}, sort_keys=False))
        print(f"wrote {len(entries)} static colliders to {out}")
    else:
        for e in entries:
            extra = f"  {e['triangles']} triangles" if e["kind"] == "mesh" else ""
            print(
                f"{nearest_distance_m(e):6.3f} m  {e['kind']:4s} "
                f"half={[round(v, 3) for v in e['half_extents']]}  "
                f"at={[round(v, 3) for v in e['xyz']]}  {e['source']}{extra}"
            )
        print(f"{len(entries)} static colliders")
    return 0


if __name__ == "__main__":
    sys.exit(main())

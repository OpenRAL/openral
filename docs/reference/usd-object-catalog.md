# USD object catalog

A deploy scene can declare objects and workspaces authored in the external USD
stage that already owns the cell or restock assets. The catalog is optional.
`deploy` forwards the declarations to the object tracker, so a scene that
declares one needs the tracker on.

**Sim only.** `DeployScene.object_catalog` requires `runtime.object_tracking` to
be `tracking` or `on`, and `openral deploy run` refuses any value other than
`off` on a real HAL. So a catalog works under `openral deploy sim`, not on real
hardware. The lower-level `load_usd_object_catalog` API stays usable on its own
for validation or inspection, without a tracker.

## Quick start

Install the optional USD reader:

```bash
just sync --group scene-usd
```

Add the catalog and the tracker settings to the `DeployScene` YAML:

```yaml
runtime:
  object_tracking: tracking
  grounding_wait_s: 3.0        # required whenever object_tracking is not off

object_catalog:
  usd: "usd-object-catalog-example.usda"   # relative to this DeployScene YAML
  prim_path: "/Cell/Catalog"               # or "/" for root-layer customLayerData
  expected_revision: "tutorial-v1"
  # reader_python: "/path/to/python-with-pxr"   # optional absolute executable path
```

Author the catalog JSON on the designated prim in USD Python:

```python
prim.SetCustomDataByKey("openral:object_catalog_json", catalog_json)
# or, for prim_path "/":
layer.customLayerData = {"openral": {"object_catalog_json": catalog_json}}
```

The [example stage](../tutorials/deploy/usd-object-catalog-example.usda) shows
the exact USDA encoding (a nested `openral` dictionary holding
`object_catalog_json`).

## Fields

| Field | Meaning |
| --- | --- |
| `object_catalog.usd` | USD file, relative to the DeployScene YAML. Keep cell and restock assets in their existing external USD; do not copy them into this repo. |
| `object_catalog.prim_path` | Prim carrying the customData value, or `/` for the root layer's `customLayerData`. |
| `object_catalog.expected_revision` | Must equal the catalog's `source_revision`. |
| `object_catalog.reader_python` | Optional executable whose Python can import `pxr.Usd`, for hosts that cannot install `usd-core` (for example a wrapper around Isaac Sim's bundled Python). OpenRAL runs only the packaged metadata reader with a timeout and starts no simulation app. This setting lives in the scene YAML, never in USD customData. |

The catalog JSON has `schema_version: 1`, a non-empty `source_revision`,
`units: "m"` and a named `stage_frame_id`. Its `objects` and `workspaces` refer
to existing USD prim paths. Names and prim paths are unique across both record
types.

- **Objects** may declare a local analytic shape. `dimension_semantics` is
  required: `nominal` (representative dimensions) or `maximum` (an upper bound).
- **Workspaces** declare a bounded box, a role (`search` or `workspace`), a
  location (`fixed` or `movable`) and a pose in the declared stage frame.

Dimensions and poses are authored metadata. The loader checks the records and
that the prims exist; it does not read USD xforms, physics state or live
simulator poses. `maximum` and `nominal` are preserved explicitly, and neither
changes collision allowance or safety behaviour.

The core schema caps catalog size and dimensions and rejects extra fields,
non-finite values, unresolved references, duplicate names or paths, and revision
mismatches. Keep the catalog prim and every referenced prim outside unloaded
payloads: the reader opens USD with payload loading disabled and refuses any
prim it cannot resolve in that view.

The catalog JSON reaches the object tracker as a bounded ROS launch parameter.
The host running the deploy needs the external USD file and either the
`scene-usd` group or a configured `reader_python`.

`DeployRuntime.object_surface_model` separately configures bounded,
observation-only surface registration. It requires every resource and
convergence bound explicitly, does not use declared dimensions to initialize or
move an object, and does not change safety checks. A valid sensor pose or
association conflict stays unresolved rather than being decided by the catalog.

## Calibrated size contradiction checks

An optional catalog `measurement_calibration` block declares
`point_error_bound_m`, `calibration_reference`, `sensor_ids` and `max_age_s`.
Every field is required when the block is present. The referenced independent
calibration must bound attributed object surface points, including depth,
extrinsics and segmentation outliers; a number in USD does not establish it.

The tracker can reject a `maximum`-size declaration when an observed lower bound
on diameter exceeds the primitive's maximum diameter plus twice the point error.
A box's maximum diameter is its diagonal; a capsule's is the segment length plus
two radii. Axis spans of measured surface points are valid lower bounds; a fitted
containing-box diagonal is not.

These do not establish a contradiction: nominal dimensions, missing calibration,
unsupported sensors, future or stale observations, occlusion and missing depth.
The check cannot detect every wrong shape.

A unique normalized label on both the catalog and observed sides associates a
candidate; repeated labels stay ambiguous. A size mismatch stays latched across
track lifecycle epochs and compatible partial observations. Clearing it would
need independent, complete, calibrated upper-bound evidence, which no producer
supplies today, and a corrected catalog revision must be independently
established. Compatibility does not prove containment, justify shrinking a held
object, or permit contact, and no authored prim pose is treated as a live
measured object pose.

## See also

- [Run a deployment and open the dashboard](../tutorials/deploy/deploy-run-and-dashboard.md)

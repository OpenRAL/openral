# External USD object declarations

Deploy scenes may refer to a catalog authored in the external USD that already owns the cell or restock assets. The catalog is optional and has the same contract for `openral deploy sim` and `openral deploy run`.

Because deploy forwards declarations to the object tracker, a scene that declares a catalog must enable tracking and provide its required grounding wait:

```yaml
runtime:
  object_tracking: shadow
  grounding_wait_s: 3.0
```

The lower-level `load_usd_object_catalog` API remains usable on its own for validation or inspection without a tracker.

```yaml
object_catalog:
  usd: "usd-object-catalog-example.usda" # relative to this DeployScene YAML
  prim_path: "/Cell/Catalog" # or "/" for root-layer customLayerData
  expected_revision: "tutorial-v1"
  # reader_python: "/path/to/python-with-pxr" # optional; absolute executable path
```

When the deploy host cannot install `usd-core` for its Python version or architecture, set
`reader_python` to a separately provisioned executable whose Python environment can import
`pxr.Usd` (for example, a wrapper that configures Isaac Sim's bundled Python and USD library
paths). OpenRAL runs only the packaged metadata reader with a timeout; it does not start a
simulation app. This setting belongs to the DeployScene YAML and is never read from or authored
into USD customData. Without it, the loader keeps using the host's in-process `pxr.Usd`
installation.

For a designated prim, `prim_path` identifies the prim carrying a string customData value at the namespaced key `openral:object_catalog_json`. In USD Python, author it with `prim.SetCustomDataByKey("openral:object_catalog_json", catalog_json)`. USDA writes that namespaced value as a nested `openral` dictionary containing `object_catalog_json`; [the example stage](usd-object-catalog-example.usda) shows the exact encoding. Set `prim_path: "/"` to store the same value in the root layer's `customLayerData` dictionary instead: `layer.customLayerData = {"openral": {"object_catalog_json": catalog_json}}`. Keep the catalog prim and all referenced object/workspace prims outside unloaded payloads: the reader opens USD with payload loading disabled and refuses any declaration prim it cannot resolve in that view. The catalog JSON has `schema_version: 1`, a non-empty `source_revision`, `units: "m"`, and a named `stage_frame_id`. `source_revision` must equal `expected_revision`. Its `objects` and `workspaces` refer to existing USD prim paths. Names and prim paths are unique across both record types.

Each object may declare a local analytic shape. Its `dimension_semantics` is required and is either `nominal` (authored representative dimensions) or `maximum` (authored upper bound). Workspace records declare a bounded box, role (`search` or `workspace`), location (`fixed` or `movable`), and pose in the declared stage frame. Catalog dimensions and poses are authored metadata: the loader checks records and prim existence, but does not read USD xforms, physics state, or live simulator poses. `maximum` and `nominal` are preserved explicitly; neither currently changes collision allowance or tightens safety behavior.

The core schema caps catalog size and dimensions and rejects extra fields, non-finite values, unresolved references, duplicate names/paths, and revision mismatches. The standalone loader imports only `pxr.Usd`; install its optional dependency with:

```bash
just sync --group scene-usd
```

Catalog JSON is passed as a bounded ROS launch parameter to the object tracker. The host running `deploy run` needs the external USD file and either the `scene-usd` dependency or a configured `reader_python` that can import `pxr.Usd`. Keep cell/restock asset content in its existing external USD; do not copy it into this repository.

`DeployRuntime.object_surface_model` separately configures bounded, observation-only surface registration. It requires all resource and convergence bounds explicitly. It does not use declared dimensions to initialize or move an object, and it does not change safety checks. A future consumer must keep a valid sensor pose/association conflict unresolved rather than preferring the catalog.


## Calibrated size contradiction checks

An optional catalog `measurement_calibration` declares `point_error_bound_m`,
`calibration_reference`, `sensor_ids`, and `max_age_s`. Every field is required
when the block is present. The referenced independent calibration must bound
attributed object surface points, including depth, extrinsics and segmentation
outliers. A number in USD does not establish that calibration.

The tracker can reject a maximum-size declaration when an observed lower bound
on diameter exceeds the primitive's maximum diameter plus twice the point error.
A box's maximum diameter is its diagonal; a capsule's is segment length plus two
radii. Axis spans of measured surface points are valid lower bounds; a fitted
containing box diagonal is not. Nominal dimensions, missing calibration,
unsupported sensors, future/stale observations, occlusion and missing depth do
not establish contradiction. This check cannot detect every wrong shape.

A unique normalized label on both catalog and observed sides associates a
candidate. Repeated labels remain ambiguous. Size mismatch stays latched across
track lifecycle epochs and compatible partial observations. Clearing it would
require independent complete calibrated upper-bound evidence; that producer is
not implemented. A corrected catalog revision must be independently established.
Compatibility does not prove containment, justify held
object shrinking, or permit contact. No authored prim pose is treated as a live
measured object pose.

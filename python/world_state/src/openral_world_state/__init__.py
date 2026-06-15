"""openral World State — snapshot aggregator + persistent spatial memory public API.

Public surface:
- ``WorldStateAggregator``: tf2-aware, injectable aggregator that produces
  ``WorldState`` Pydantic snapshots at up to 30 Hz.
- ``DEFAULT_RATE_HZ``: Advertised default snapshot rate.
- ``DEFAULT_STALENESS_S``: Default staleness threshold in seconds.
- ``SpatialMemory``: ADR-0038 persistent object-centric scene-graph memory —
  accumulates ``WorldState.detected_objects`` and answers ``RecallObjectQuery`` /
  ``ResolvePlaceQuery``.
- ``compute_approach_viewpoint``: camera-facing standoff-pose helper.
- ``look_at_quat_wxyz`` / ``compute_gaze_pose``: ADR-0044 shared gaze geometry
  (look-at rotations per camera convention; the ``rskill-moveit-look-at`` goal pose).
- ``OccupancyGridIndex`` / ``refine_approach_pose``: ADR-0044 occupancy-grid
  queries + approach-pose snapping (free cell + line-of-sight; planning-layer,
  not a safety surface).
- ``OpenClipEmbedder`` / ``TextEmbedder``: ADR-0038 §5 open-vocab text embedder
  (OpenCLIP ViT-B/32, MIT) — optional, ``uv sync --group clip``.
- ``emit_scene_objects_span`` / ``scene_objects_payload``: publish the remembered
  object nodes as a ``world.scene_objects`` OTel span for the dashboard.
- ``VoxelFrustumLifter`` / ``ObjectMemory`` / lift helpers (ADR-0035):
  lift 2D detections to 3D object centres and remember them with IoU gating.
"""

from openral_world_state.aggregator import (
    DEFAULT_RATE_HZ,
    DEFAULT_STALENESS_S,
    WorldStateAggregator,
)
from openral_world_state.embedder import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    OpenClipEmbedder,
    TextEmbedder,
)
from openral_world_state.geometry import (
    ViewAxis,
    compute_gaze_pose,
    look_at_quat_wxyz,
    rotation_to_quat_wxyz,
)
from openral_world_state.grid import (
    FREE_MAX,
    OccupancyGridIndex,
    refine_approach_pose,
)
from openral_world_state.object_lift import (
    VoxelFrustumLifter,
    aabb_iou_3d,
    build_in_fov_predicate,
    decode_occupied_centers,
    depth_cloud_to_centers_base,
    homogeneous_from_quat_xyz,
)
from openral_world_state.object_memory import ObjectMemory
from openral_world_state.scene_objects_span import (
    emit_scene_objects_span,
    scene_objects_payload,
)
from openral_world_state.spatial_memory import (
    DEFAULT_ASSOC_DISTANCE_M,
    DEFAULT_CAMERA_FRAME,
    DEFAULT_MAP_FRAME,
    DEFAULT_MIN_TEXT_SIMILARITY,
    DEFAULT_STANDOFF_M,
    SpatialMemory,
    compute_approach_viewpoint,
)

__all__ = [
    "DEFAULT_ASSOC_DISTANCE_M",
    "DEFAULT_CAMERA_FRAME",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_CLIP_PRETRAINED",
    "DEFAULT_MAP_FRAME",
    "DEFAULT_MIN_TEXT_SIMILARITY",
    "DEFAULT_RATE_HZ",
    "DEFAULT_STALENESS_S",
    "DEFAULT_STANDOFF_M",
    "FREE_MAX",
    "ObjectMemory",
    "OccupancyGridIndex",
    "OpenClipEmbedder",
    "SpatialMemory",
    "TextEmbedder",
    "ViewAxis",
    "VoxelFrustumLifter",
    "WorldStateAggregator",
    "aabb_iou_3d",
    "build_in_fov_predicate",
    "compute_approach_viewpoint",
    "compute_gaze_pose",
    "decode_occupied_centers",
    "depth_cloud_to_centers_base",
    "emit_scene_objects_span",
    "homogeneous_from_quat_xyz",
    "look_at_quat_wxyz",
    "refine_approach_pose",
    "rotation_to_quat_wxyz",
    "scene_objects_payload",
]
__version__ = "0.1.0"

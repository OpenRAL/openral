// SPDX-License-Identifier: Apache-2.0
// When the bridge may still publish the octree it last received.
//
// `octomap_server` publishes only when it inserts a cloud, so a dead camera is
// a silent `/octomap_binary`. The bridge re-rasterizes its last octree on a
// timer (so the grid follows the robot through TF between octrees); past this
// bound it must stop, or the safety kernel — which times voxel freshness from
// receipt — sees a frozen map arrive on time and never fails closed. Silence
// is the signal: the kernel's own `world_voxel_deadline_ms` turns it into
// `DROP_VOXEL_UNAVAILABLE`. Hazard log Entry 033.

#pragma once

#include <cmath>

namespace openral_octomap_bridge {

/// Hard cap on `max_octree_age_s`: the safety kernel's own cap on
/// `world_voxel_deadline_ms` (2000 ms), which the bound must not exceed. Mirrors
/// `openral_core.DeployRuntime.world_voxel_deadline_s <= 2.0`;
/// `tests/unit/test_perception_caps_mirror.py` pins it.
inline constexpr double kMaxOctreeAgeS = 2.0;

/// Is `max_age_s` a usable `max_octree_age_s`? In (0, kMaxOctreeAgeS]; NaN and
/// infinity are not.
inline bool valid_max_octree_age(double max_age_s) {
  return max_age_s > 0.0 && max_age_s <= kMaxOctreeAgeS;
}

/// May an octree received `age_s` seconds ago (receipt time, on the node's
/// clock — the same clock the kernel times voxel freshness on) still be
/// published? False for an unusable bound, and for an age the clock cannot
/// explain (negative or non-finite, e.g. a sim clock reset): a map of unknown
/// age is not fresh.
inline bool octree_is_fresh(double age_s, double max_age_s) {
  return valid_max_octree_age(max_age_s) && std::isfinite(age_s) && age_s >= 0.0 &&
         age_s <= max_age_s;
}

}  // namespace openral_octomap_bridge

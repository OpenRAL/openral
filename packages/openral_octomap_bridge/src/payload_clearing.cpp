// SPDX-License-Identifier: Apache-2.0
// Attach-time (and ongoing) clearing of a grasped payload's own occupancy.

#include "openral_octomap_bridge/payload_clearing.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <openral_msgs/msg/attached_collision_primitive.hpp>

namespace openral_octomap_bridge {

namespace {

using Wire = openral_msgs::msg::AttachedCollisionPrimitive;

bool finite_positive(double v) { return std::isfinite(v) && v >= 0.0; }

// A wire pose → tf2, refusing a degenerate quaternion rather than silently
// treating it as the identity (the bridge would then clear the wrong volume).
bool pose_from_msg(const geometry_msgs::msg::Pose& pose, tf2::Transform& out) {
  const double norm2 =
      pose.orientation.x * pose.orientation.x + pose.orientation.y * pose.orientation.y +
      pose.orientation.z * pose.orientation.z + pose.orientation.w * pose.orientation.w;
  if (!std::isfinite(norm2) || norm2 < 1e-12) {
    return false;
  }
  if (!std::isfinite(pose.position.x) || !std::isfinite(pose.position.y) ||
      !std::isfinite(pose.position.z)) {
    return false;
  }
  tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
  q.normalize();
  out = tf2::Transform(q, tf2::Vector3(pose.position.x, pose.position.y, pose.position.z));
  return true;
}

// Signed distance from a point to the primitive's surface (negative inside).
double surface_distance(const PayloadPrimitive& prim, const tf2::Vector3& local) {
  switch (prim.shape_type) {
  case Wire::SHAPE_SPHERE:
    return local.length() - prim.radius;
  case Wire::SHAPE_CAPSULE: {
    // The central segment runs along the primitive's local +Z.
    const double z = std::clamp(local.z(), -prim.half_length, prim.half_length);
    return tf2::Vector3(local.x(), local.y(), local.z() - z).length() - prim.radius;
  }
  case Wire::SHAPE_BOX:
  default: {
    const double ox = std::fabs(local.x()) - prim.half_extents.x();
    const double oy = std::fabs(local.y()) - prim.half_extents.y();
    const double oz = std::fabs(local.z()) - prim.half_extents.z();
    // Outside distance only; a point inside reports 0, which is <= any
    // non-negative bound and therefore still clears.
    return tf2::Vector3(std::max(ox, 0.0), std::max(oy, 0.0), std::max(oz, 0.0)).length();
  }
  }
}

// Radius of a sphere at the primitive's origin that contains it.
double bounding_radius(const PayloadPrimitive& prim) {
  switch (prim.shape_type) {
  case Wire::SHAPE_SPHERE:
    return prim.radius;
  case Wire::SHAPE_CAPSULE:
    return prim.radius + prim.half_length;
  case Wire::SHAPE_BOX:
  default:
    return prim.half_extents.length();
  }
}

}  // namespace

bool place_attached_object(const openral_msgs::msg::AttachedCollisionObject& object,
                           const tf2::Transform& grid_from_link, std::vector<PayloadPrimitive>& out,
                           std::vector<SupportPatch>& out_patches) {
  if (object.primitives.empty()) {
    return false;
  }
  tf2::Transform link_from_object;
  if (!pose_from_msg(object.pose_in_link, link_from_object)) {
    return false;
  }
  const tf2::Transform grid_from_object = grid_from_link * link_from_object;

  // The support-contact attestation, validated by the kernel's own ingest
  // rules (`ingest_attached_objects`): a producer that cannot describe its own
  // support contact is one whose payload volume this bridge does not trust
  // either, so a malformed witness rejects the whole object rather than
  // degrading to "no witness" — which would silently clear the support surface.
  bool has_patch = false;
  SupportPatch patch;
  if (object.support_contact_valid) {
    const auto& wire = object.support_contact;
    const tf2::Vector3 normal_in(wire.contact_normal_in_object.x, wire.contact_normal_in_object.y,
                                 wire.contact_normal_in_object.z);
    const double norm2 = normal_in.length2();
    if (!std::isfinite(norm2) || norm2 <= 1e-12 || !std::isfinite(wire.contact_point_in_object.x) ||
        !std::isfinite(wire.contact_point_in_object.y) ||
        !std::isfinite(wire.contact_point_in_object.z) || !std::isfinite(wire.patch_radius_m) ||
        wire.patch_radius_m <= 0.0 || !std::isfinite(wire.max_penetration_m) ||
        wire.max_penetration_m < 0.0) {
      return false;
    }
    const tf2::Vector3 point_in(wire.contact_point_in_object.x, wire.contact_point_in_object.y,
                                wire.contact_point_in_object.z);
    patch.point = grid_from_object * point_in;
    // The normal is a direction: rotate it, never translate it.
    patch.normal = (grid_from_object.getBasis() * normal_in).normalized();
    patch.patch_radius = wire.patch_radius_m;
    patch.max_penetration = wire.max_penetration_m;
    has_patch = true;
  }

  // Staged locally so a rejection late in the list appends nothing at all.
  std::vector<PayloadPrimitive> staged;
  staged.reserve(object.primitives.size());
  for (const auto& wire : object.primitives) {
    PayloadPrimitive prim;
    prim.shape_type = wire.shape_type;
    const std::size_t n_dims = wire.shape_dimensions.size();
    if (wire.shape_type == Wire::SHAPE_SPHERE) {
      if (n_dims < 1 || !finite_positive(wire.shape_dimensions[0])) {
        return false;
      }
      prim.radius = wire.shape_dimensions[0];
    } else if (wire.shape_type == Wire::SHAPE_CAPSULE) {
      if (n_dims < 2 || !finite_positive(wire.shape_dimensions[0]) ||
          !finite_positive(wire.shape_dimensions[1])) {
        return false;
      }
      prim.radius = wire.shape_dimensions[0];
      // The wire carries the FULL central-segment length; this uses halves,
      // exactly as the kernel's ingest does.
      prim.half_length = 0.5 * wire.shape_dimensions[1];
    } else if (wire.shape_type == Wire::SHAPE_BOX) {
      if (n_dims < 3 || !finite_positive(wire.shape_dimensions[0]) ||
          !finite_positive(wire.shape_dimensions[1]) ||
          !finite_positive(wire.shape_dimensions[2])) {
        return false;
      }
      prim.half_extents = tf2::Vector3(wire.shape_dimensions[0], wire.shape_dimensions[1],
                                       wire.shape_dimensions[2]);
    } else {
      // Unknown tag: the kernel fails the message closed, and the bridge must
      // not guess at a volume it cannot describe.
      return false;
    }
    tf2::Transform object_from_primitive;
    if (!pose_from_msg(wire.pose_in_object, object_from_primitive)) {
      return false;
    }
    prim.pose = grid_from_object * object_from_primitive;
    staged.push_back(prim);
  }
  out.insert(out.end(), staged.begin(), staged.end());
  if (has_patch) {
    out_patches.push_back(patch);
  }
  return true;
}

bool support_patch_withholds(const SupportPatch& patch, const tf2::Vector3& center,
                             double resolution) noexcept {
  if (!(patch.patch_radius > 0.0) || !(resolution > 0.0)) {
    return false;
  }
  const tf2::Vector3 delta = center - patch.point;
  const double height = delta.dot(patch.normal);
  const double lateral_sq = std::max(0.0, delta.length2() - height * height);
  const double half = 0.5 * resolution;
  // Both pads are exact discretisation geometry, not tuned tolerances, and are
  // the kernel's own: the cube's half-width projected on the normal, and the
  // cube's circumradius laterally.
  const double normal_half_width =
      half *
      (std::fabs(patch.normal.x()) + std::fabs(patch.normal.y()) + std::fabs(patch.normal.z()));
  const double lateral_pad = half * 1.7320508075688772;  // sqrt(3)
  const double reach = patch.patch_radius + lateral_pad;
  if (lateral_sq > reach * reach) {
    return false;
  }
  // One voxel of co-planar headroom, mirroring the kernel's 2026-08-15
  // calibration (hazard log Entry 012, "Calibration 2026-08-15") term for term.
  // It is added HERE as well, and in lockstep, because `withheld ⊆ exempt` is a
  // property of the two predicates being the same inequality — the moment the
  // kernel's bound moves and this one does not, the containment stops holding by
  // construction and starts holding by luck.
  //
  // No `slack` term (the kernel's `attached_contact_tolerance`): what this
  // withholds must stay a subset of what the kernel exempts.
  return height <= normal_half_width + patch.max_penetration + resolution;
}

std::size_t clear_attached_payload_cells(openral_msgs::msg::OccupancyVoxels& grid,
                                         const std::vector<PayloadPrimitive>& primitives,
                                         const std::vector<SupportPatch>& patches,
                                         double padding_m) {
  if (primitives.empty() || !(grid.resolution > 0.0)) {
    return 0;
  }
  const auto sx = static_cast<long>(grid.size_x);
  const auto sy = static_cast<long>(grid.size_y);
  const auto sz = static_cast<long>(grid.size_z);
  const std::size_t n_cells = static_cast<std::size_t>(grid.size_x) *
                              static_cast<std::size_t>(grid.size_y) *
                              static_cast<std::size_t>(grid.size_z);
  if (n_cells == 0 || grid.occupancy.size() != n_cells) {
    return 0;  // a grid that does not describe itself is not one to edit
  }

  // The cell cube's circumradius: every cell the payload's volume intersects
  // has its centre within it. Extra padding beyond that is protection given up
  // deliberately by the caller (see the header).
  const double pad = (std::isfinite(padding_m) && padding_m > 0.0) ? padding_m : 0.0;
  const double reach = 0.5 * grid.resolution * std::sqrt(3.0) + pad;
  const double res = grid.resolution;

  std::size_t cleared = 0;
  for (const auto& prim : primitives) {
    const double span = bounding_radius(prim) + reach;
    const tf2::Vector3 origin = prim.pose.getOrigin();
    if (!std::isfinite(span) || !std::isfinite(origin.x()) || !std::isfinite(origin.y()) ||
        !std::isfinite(origin.z())) {
      continue;
    }
    // Only the cells inside the primitive's inflated AABB can qualify, so the
    // cost is the payload's own volume rather than the grid's.
    const double lo[3] = {origin.x() - span - grid.origin.x, origin.y() - span - grid.origin.y,
                          origin.z() - span - grid.origin.z};
    const double hi[3] = {origin.x() + span - grid.origin.x, origin.y() + span - grid.origin.y,
                          origin.z() + span - grid.origin.z};
    const long size[3] = {sx, sy, sz};
    long first[3];
    long last[3];
    bool empty_window = false;
    for (int axis = 0; axis < 3; ++axis) {
      first[axis] = std::max<long>(0, static_cast<long>(std::ceil(lo[axis] / res - 0.5)));
      last[axis] =
          std::min<long>(size[axis] - 1, static_cast<long>(std::floor(hi[axis] / res - 0.5)));
      empty_window = empty_window || first[axis] > last[axis];
    }
    if (empty_window) {
      continue;
    }

    const tf2::Transform primitive_from_grid = prim.pose.inverse();
    for (long iz = first[2]; iz <= last[2]; ++iz) {
      for (long iy = first[1]; iy <= last[1]; ++iy) {
        for (long ix = first[0]; ix <= last[0]; ++ix) {
          const auto idx = static_cast<std::size_t>(ix + sx * (iy + sy * iz));
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          const tf2::Vector3 center(grid.origin.x + (static_cast<double>(ix) + 0.5) * res,
                                    grid.origin.y + (static_cast<double>(iy) + 0.5) * res,
                                    grid.origin.z + (static_cast<double>(iz) + 0.5) * res);
          if (surface_distance(prim, primitive_from_grid * center) > reach) {
            continue;
          }
          // The partition: a cell an attested support patch claims is the
          // supporting surface, not the payload. It stays in the map for the
          // kernel's witness to keep measuring against — and for Nav2 and SLAM,
          // whose obstacle it also is. Every object's attestation guards every
          // object's clearing; a second payload's volume must not erase the
          // first one's support either.
          bool withheld = false;
          for (const auto& patch : patches) {
            if (support_patch_withholds(patch, center, res)) {
              withheld = true;
              break;
            }
          }
          if (!withheld) {
            grid.occupancy[idx] = 0;
            ++cleared;
          }
        }
      }
    }
  }
  return cleared;
}

std::vector<std::uint8_t>
AttachSweepLedger::sweep(const std::vector<AttachSweepObservation>& present, double window_m) {
  const double window = (std::isfinite(window_m) && window_m > 0.0) ? window_m : 0.0;
  std::vector<Window> next;
  next.reserve(present.size());
  std::vector<std::uint8_t> open;
  open.reserve(present.size());

  for (const auto& observation : present) {
    // A key seen for the first time anchors HERE, so its displacement is 0 and
    // its window is open: that grid is the attach transition itself.
    Window window_state{observation.key, observation.payload_position, false};
    for (const auto& prior : windows_) {
      if (prior.key.attachment_revision == observation.key.attachment_revision &&
          prior.key.object_id == observation.key.object_id) {
        window_state = prior;
        break;
      }
    }
    // Closing latches: a payload that wanders back to where it was grasped does
    // not get the widened reach again, because the map around it by then is
    // evidence gathered while the payload was somewhere else.
    if (!window_state.closed &&
        (observation.payload_position - window_state.anchor).length() > window) {
      window_state.closed = true;
    }
    open.push_back(window_state.closed ? 0U : 1U);
    next.push_back(window_state);
  }

  windows_ = std::move(next);
  return open;
}

std::size_t AttachSweepLedger::size() const noexcept { return windows_.size(); }

double attach_transition_padding(double steady_padding_m, double attach_sweep_padding_m) noexcept {
  const double extra = (std::isfinite(attach_sweep_padding_m) && attach_sweep_padding_m > 0.0)
                           ? attach_sweep_padding_m
                           : 0.0;
  return (std::isfinite(steady_padding_m) && steady_padding_m > 0.0 ? steady_padding_m : 0.0) +
         extra;
}

}  // namespace openral_octomap_bridge

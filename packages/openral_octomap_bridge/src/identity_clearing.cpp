// SPDX-License-Identifier: Apache-2.0
// Octomap clearing by identity (issue #418).

#include "openral_octomap_bridge/identity_clearing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>

#include <openral_msgs/msg/attached_collision_primitive.hpp>

namespace openral_octomap_bridge {

namespace {

using Wire = openral_msgs::msg::AttachedCollisionPrimitive;
constexpr double kSqrt3 = 1.7320508075688772;

bool finite(const tf2::Vector3& v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

// The kernel's tight-geometry shape rules, read fail-closed.
bool parse_tight(const Wire& wire, IdentityPiece& piece) {
  const auto& lo = wire.tight_dop_lo;
  const auto& hi = wire.tight_dop_hi;
  const auto& v = wire.tight_hull_vertices;
  if (lo.empty() && hi.empty() && v.empty()) {
    return true;  // no refinement: the box is the model
  }
  if (wire.shape_type != Wire::SHAPE_BOX || lo.size() != kSelfDopAxes ||
      hi.size() != kSelfDopAxes || v.size() % 3 != 0 ||
      v.size() / 3 > static_cast<std::size_t>(kSelfMaxHullVertices)) {
    return false;
  }
  for (int k = 0; k < kSelfDopAxes; ++k) {
    const auto i = static_cast<std::size_t>(k);
    if (!std::isfinite(lo[i]) || !std::isfinite(hi[i]) || lo[i] > hi[i]) {
      return false;
    }
    piece.tight.dop_lo[k] = lo[i];
    piece.tight.dop_hi[k] = hi[i];
  }
  piece.tight.vertices.clear();
  for (std::size_t i = 0; i < v.size(); i += 3) {
    const tf2::Vector3 p(v[i], v[i + 1], v[i + 2]);
    if (!finite(p)) {
      return false;
    }
    piece.tight.vertices.push_back(p);
  }
  piece.has_tight = true;
  return true;
}

}  // namespace

bool tracked_set_usable(const openral_msgs::msg::WorldStateStamped& state,
                        const std::string& grid_frame) {
  if (!state.tracked_objects_valid || state.tracked_objects.frame_id != grid_frame) {
    return false;
  }
  return std::all_of(state.tracked_objects.objects.begin(), state.tracked_objects.objects.end(),
                     [&](const auto& t) { return t.box.frame_id == grid_frame; });
}

const openral_msgs::msg::TrackedObject*
identity_clearing_track(const openral_msgs::msg::AttachedCollisionObject& object,
                        const openral_msgs::msg::WorldStateStamped& state,
                        const std::string& grid_frame) {
  if (object.evidence_kind != kTrackedObjectEvidence || !tracked_set_usable(state, grid_frame)) {
    return nullptr;
  }
  for (const auto& other : state.attached_objects) {
    if (&other != &object && other.attach_link == object.attach_link) {
      return nullptr;  // two payloads on one hand: whose cells are whose is not known
    }
  }
  const openral_msgs::msg::TrackedObject* match = nullptr;
  for (const auto& track : state.tracked_objects.objects) {
    if (track.object_id == object.object_id) {
      if (match != nullptr) {
        return nullptr;  // a duplicated id is no identity
      }
      match = &track;
    } else if (track.status == "bound" && track.attach_link == object.attach_link) {
      return nullptr;  // a second object bound to this hand: ambiguous
    }
  }
  if (match == nullptr || match->status != "bound" || match->epoch != object.epoch ||
      match->attach_link != object.attach_link || !std::isfinite(match->floor_z)) {
    return nullptr;
  }
  return match;
}

std::vector<const openral_msgs::msg::TrackedObject*>
advance_release_clears(ReleaseClears& windows, const IdentityKeys& was_bound,
                       const openral_msgs::msg::WorldStateStamped& state,
                       const std::string& grid_frame, std::int64_t now_ns, double timeout_s) {
  std::vector<const openral_msgs::msg::TrackedObject*> out;
  if (!tracked_set_usable(state, grid_frame)) {
    windows.clear();
    return out;
  }
  const auto timeout_ns = static_cast<double>(timeout_s) * 1e9;
  ReleaseClears kept;
  for (const auto& track : state.tracked_objects.objects) {
    if (track.status != "released") {
      continue;
    }
    const auto open = windows.find(track.object_id);
    ReleaseClear w;
    if (open != windows.end() && open->second.epoch == track.epoch) {
      w = open->second;
    } else if (was_bound.count({track.object_id, track.epoch}) != 0) {
      w = ReleaseClear{track.epoch, now_ns};
    } else {
      continue;
    }
    const auto same_id = [&](const auto& x) { return x.object_id == track.object_id; };
    const auto& objs = state.tracked_objects.objects;
    const auto& wire = state.attached_objects;
    const auto bound = std::find_if(wire.begin(), wire.end(), same_id);
    PayloadPrimitive box;
    if (std::count_if(objs.begin(), objs.end(), same_id) != 1 ||
        std::count_if(wire.begin(), wire.end(), same_id) != 1 ||
        bound->evidence_kind != kTrackedObjectEvidence || bound->epoch != track.epoch ||
        !std::isfinite(track.floor_z) || !track_box(track, box) ||
        static_cast<double>(now_ns - w.opened_ns) > timeout_ns) {
      continue;  // closed: erased with the rest below
    }
    kept[track.object_id] = w;
    out.push_back(&track);
  }
  windows = std::move(kept);
  return out;
}

bool track_box(const openral_msgs::msg::TrackedObject& track, PayloadPrimitive& out) {
  const auto& h = track.box.half_extents;
  if (!(h.x > 0.0) || !(h.y > 0.0) || !(h.z > 0.0) || !std::isfinite(h.x) || !std::isfinite(h.y) ||
      !std::isfinite(h.z)) {
    return false;
  }
  PayloadPrimitive box;
  box.shape_type = Wire::SHAPE_BOX;
  box.half_extents = tf2::Vector3(h.x, h.y, h.z);
  if (!pose_from_msg(track.box.pose, box.pose)) {
    return false;
  }
  out = box;
  return true;
}

bool place_identity_pieces(const openral_msgs::msg::AttachedCollisionObject& object,
                           const tf2::Transform& frame_from_link, std::vector<IdentityPiece>& out) {
  out.clear();
  std::vector<PayloadPrimitive> prims;
  std::vector<SupportPatch> patches;
  if (!place_attached_object(object, frame_from_link, prims, patches)) {
    return false;
  }
  for (std::size_t i = 0; i < prims.size(); ++i) {
    IdentityPiece piece;
    piece.prim = prims[i];
    piece.local_from_frame = prims[i].pose.inverse();
    if (!parse_tight(object.primitives[i], piece)) {
      out.clear();
      return false;
    }
    out.push_back(std::move(piece));
  }
  return true;
}

bool identity_covers(const std::vector<IdentityPiece>& pieces, const tf2::Vector3& c,
                     double half_cell) {
  for (const auto& p : pieces) {
    if ((c - p.prim.pose.getOrigin()).length() > primitive_bounding_radius(p.prim) + half_cell) {
      continue;
    }
    const tf2::Vector3 local = p.local_from_frame * c;
    if (primitive_surface_distance(p.prim, local) > half_cell) {
      continue;
    }
    // Stops as soon as a hull point within half a cell is found (certainly covered) or
    // the bound passes it (certainly not).
    if (p.has_tight && self_hull_distance(p.tight, local, half_cell, half_cell) > half_cell) {
      continue;
    }
    return true;
  }
  return false;
}

bool identity_keep_out(const openral_msgs::msg::TrackedObjectSet& set,
                       const openral_msgs::msg::TrackedObject& self,
                       const tf2::Transform& frame_from_grid, bool keep_own_box,
                       IdentityKeepOut& out) {
  out = IdentityKeepOut{};
  out.floor_point = frame_from_grid * tf2::Vector3(0.0, 0.0, self.floor_z);
  out.floor_normal = frame_from_grid.getBasis() * tf2::Vector3(0.0, 0.0, 1.0);
  if (!finite(out.floor_point)) {
    return false;
  }
  for (const auto& track : set.objects) {
    if (track.object_id == self.object_id && !keep_own_box) {
      continue;
    }
    PayloadPrimitive box;
    if (!track_box(track, box)) {
      return false;
    }
    box.pose = frame_from_grid * box.pose;
    out.boxes.push_back(box);
  }
  return true;
}

bool released_clear_pieces(const openral_msgs::msg::TrackedObjectSet& set,
                           const openral_msgs::msg::TrackedObject& track,
                           const tf2::Transform& frame_from_grid,
                           std::vector<IdentityPiece>& pieces, IdentityKeepOut& keep) {
  pieces.clear();
  IdentityPiece piece;
  if (!track_box(track, piece.prim) ||
      !identity_keep_out(set, track, frame_from_grid, false, keep)) {
    return false;
  }
  piece.prim.pose = frame_from_grid * piece.prim.pose;
  piece.local_from_frame = piece.prim.pose.inverse();
  pieces.push_back(std::move(piece));
  return true;
}

bool identity_keeps(const IdentityKeepOut& keep, const tf2::Vector3& c, double resolution) {
  if ((c - keep.floor_point).dot(keep.floor_normal) < 0.0) {
    return true;
  }
  const double circumradius = 0.5 * resolution * kSqrt3;
  return std::any_of(keep.boxes.begin(), keep.boxes.end(), [&](const PayloadPrimitive& b) {
    return primitive_surface_distance(b, b.pose.inverse() * c) <= circumradius;
  });
}

std::size_t clear_identity_cells(openral_msgs::msg::OccupancyVoxels& grid,
                                 const std::vector<IdentityPiece>& pieces,
                                 const IdentityKeepOut& keep) {
  const double res = grid.resolution;
  const std::size_t n_cells = static_cast<std::size_t>(grid.size_x) * grid.size_y * grid.size_z;
  const tf2::Quaternion q(grid.orientation.x, grid.orientation.y, grid.orientation.z,
                          grid.orientation.w);
  if (pieces.empty() || !(res > 0.0) || n_cells == 0 || grid.occupancy.size() != n_cells ||
      std::fabs(q.length2() - 1.0) > 1e-6) {
    return 0;
  }
  const tf2::Transform base_from_grid(q, tf2::Vector3(grid.origin.x, grid.origin.y, grid.origin.z));
  const tf2::Transform grid_from_base = base_from_grid.inverse();
  const long size[3] = {static_cast<long>(grid.size_x), static_cast<long>(grid.size_y),
                        static_cast<long>(grid.size_z)};
  std::size_t cleared = 0;
  for (const auto& p : pieces) {
    const double span = primitive_bounding_radius(p.prim) + 0.5 * res;
    const tf2::Vector3 o = grid_from_base * p.prim.pose.getOrigin();
    if (!std::isfinite(span) || !finite(o)) {
      continue;
    }
    long first[3];
    long last[3];
    for (int a = 0; a < 3; ++a) {
      first[a] = std::max<long>(0, static_cast<long>(std::ceil((o[a] - span) / res - 0.5)));
      last[a] =
          std::min<long>(size[a] - 1, static_cast<long>(std::floor((o[a] + span) / res - 0.5)));
    }
    for (long iz = first[2]; iz <= last[2]; ++iz) {
      for (long iy = first[1]; iy <= last[1]; ++iy) {
        for (long ix = first[0]; ix <= last[0]; ++ix) {
          const auto idx = static_cast<std::size_t>(ix + size[0] * (iy + size[1] * iz));
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          const tf2::Vector3 c =
              base_from_grid * tf2::Vector3((static_cast<double>(ix) + 0.5) * res,
                                            (static_cast<double>(iy) + 0.5) * res,
                                            (static_cast<double>(iz) + 0.5) * res);
          if (identity_covers(pieces, c, 0.5 * res) && !identity_keeps(keep, c, res)) {
            grid.occupancy[idx] = 0;
            ++cleared;
          }
        }
      }
    }
  }
  return cleared;
}

void append_swept_poses(std::vector<tf2::Transform>& out, const tf2::Transform& a,
                        const tf2::Transform& b, double reach_m, double step_m) {
  const tf2::Quaternion qa = a.getRotation();
  const tf2::Quaternion qb = b.getRotation();
  const double travel =
      (b.getOrigin() - a.getOrigin()).length() + qa.angleShortestPath(qb) * std::max(reach_m, 0.0);
  const double n_raw = step_m > 0.0 ? std::ceil(travel / step_m) : 1.0;
  const int n = std::isfinite(n_raw) ? static_cast<int>(std::clamp(n_raw, 1.0, 1000.0)) : 1;
  for (int i = 1; i <= n; ++i) {
    const double s = static_cast<double>(i) / n;
    out.emplace_back(qa.slerp(qb, s), a.getOrigin().lerp(b.getOrigin(), s));
  }
}

std::vector<SweptBox> octree_clear_boxes(const octomap::OcTree& tree, const SweptBox& window,
                                         const std::function<bool(const tf2::Vector3&)>& clear,
                                         std::size_t max_boxes, bool& truncated) {
  truncated = false;
  std::vector<SweptBox> out;
  octomap::OcTreeKey lo_key;
  octomap::OcTreeKey hi_key;
  if (!window.valid ||
      !tree.coordToKeyChecked(octomap::point3d(static_cast<float>(window.lo.x()),
                                               static_cast<float>(window.lo.y()),
                                               static_cast<float>(window.lo.z())),
                              lo_key) ||
      !tree.coordToKeyChecked(octomap::point3d(static_cast<float>(window.hi.x()),
                                               static_cast<float>(window.hi.y()),
                                               static_cast<float>(window.hi.z())),
                              hi_key)) {
    return out;
  }
  const unsigned depth = tree.getTreeDepth();
  std::vector<octomap::OcTreeKey> keys;
  for (auto it = tree.begin_leafs_bbx(lo_key, hi_key), end = tree.end_leafs_bbx(); it != end;
       ++it) {
    if (it.getDepth() != depth || !tree.isNodeOccupied(*it)) {
      continue;
    }
    const octomap::point3d c = it.getCoordinate();
    if (clear(tf2::Vector3(c.x(), c.y(), c.z()))) {
      keys.push_back(it.getKey());
    }
  }
  // Runs along x, then runs with the same x span stacked along y.
  std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) {
    return std::tie(a[2], a[1], a[0]) < std::tie(b[2], b[1], b[0]);
  });
  struct Run {
    std::uint16_t x0, x1, y0, y1, z;
  };
  std::vector<Run> runs;
  for (const auto& k : keys) {
    if (!runs.empty() && runs.back().z == k[2] && runs.back().y0 == k[1] &&
        runs.back().x1 + 1 == k[0]) {
      runs.back().x1 = k[0];
    } else {
      runs.push_back(Run{k[0], k[0], k[1], k[1], k[2]});
    }
  }
  std::vector<Run> merged;
  std::map<std::tuple<std::uint16_t, std::uint16_t, std::uint16_t>, std::size_t> open;
  for (const auto& r : runs) {
    const auto at = open.find({r.z, r.x0, r.x1});
    if (at != open.end() && merged[at->second].y1 + 1 == r.y0) {
      merged[at->second].y1 = r.y0;
    } else {
      open[{r.z, r.x0, r.x1}] = merged.size();
      merged.push_back(r);
    }
  }
  const double quarter = 0.25 * tree.getResolution();
  for (const auto& r : merged) {
    if (out.size() >= max_boxes) {
      truncated = true;
      break;
    }
    const octomap::point3d lo = tree.keyToCoord(octomap::OcTreeKey(r.x0, r.y0, r.z));
    const octomap::point3d hi = tree.keyToCoord(octomap::OcTreeKey(r.x1, r.y1, r.z));
    SweptBox box;
    box.add(tf2::Vector3(lo.x() - quarter, lo.y() - quarter, lo.z() - quarter));
    box.add(tf2::Vector3(hi.x() + quarter, hi.y() + quarter, hi.z() + quarter));
    out.push_back(box);
  }
  return out;
}

}  // namespace openral_octomap_bridge

// SPDX-License-Identifier: Apache-2.0
// The `collision_*` parameters the deploy launch hands the safety kernel, declared on
// a node under an optional prefix: the robot self-filter reads its own robot's
// (prefix ""), the peer producer each peer's (`peer_<name>.`). One list, so a field
// added to `SelfModelParams` reaches both.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "openral_octomap_bridge/self_filter.hpp"

namespace openral_octomap_bridge {

struct DeclaredSelfModel {
  SelfModelParams params;
  std::vector<std::string> joint_names;
  /// Parallel to `joint_names`: a second spelling each dof answers to (the manifest's
  /// `sim_joint_name`, the upstream name a vendor `/joint_states` uses). Empty = none.
  std::vector<std::string> joint_aliases;
};

inline DeclaredSelfModel declare_self_model(rclcpp::Node& node, const std::string& prefix = "") {
  const auto ints = [&](const char* k) {
    return node.declare_parameter<std::vector<std::int64_t>>(prefix + k,
                                                             std::vector<std::int64_t>{});
  };
  const auto doubles = [&](const char* k) {
    return node.declare_parameter<std::vector<double>>(prefix + k, std::vector<double>{});
  };
  const auto strings = [&](const char* k) {
    return node.declare_parameter<std::vector<std::string>>(prefix + k, std::vector<std::string>{});
  };
  DeclaredSelfModel out;
  SelfModelParams& p = out.params;
  p.n_links = node.declare_parameter<std::int64_t>(prefix + "collision_n_links", 0);
  p.parent = ints("collision_parent");
  p.joint_kind = ints("collision_joint_kind");
  p.dof_index = ints("collision_dof_index");
  p.origin_xyzrpy = doubles("collision_origin_xyzrpy");
  p.axis = doubles("collision_axis");
  p.mimic_multiplier = doubles("collision_mimic_multiplier");
  p.mimic_offset = doubles("collision_mimic_offset");
  p.curve_point_first = ints("collision_curve_point_first");
  p.curve_point_count = ints("collision_curve_point_count");
  p.curve_points = doubles("collision_curve_points");
  p.link_names = strings("collision_link_names");
  p.capsule_link = ints("collision_capsule_link");
  p.capsule_radius = doubles("collision_capsule_radius");
  p.capsule_half_length = doubles("collision_capsule_half_length");
  p.capsule_origin_xyzrpy = doubles("collision_capsule_origin_xyzrpy");
  p.box_link = ints("collision_box_link");
  p.box_half_extents = doubles("collision_box_half_extents");
  p.box_origin_xyzrpy = doubles("collision_box_origin_xyzrpy");
  // The kernel's tight geometry for those boxes (a 26-DOP and, within the kernel's
  // vertex budget, the exact convex hull).
  p.box_hull = ints("collision_box_hull");
  p.hull_dop_lo = doubles("collision_hull_dop_lo");
  p.hull_dop_hi = doubles("collision_hull_dop_hi");
  p.hull_vertex_first = ints("collision_hull_vertex_first");
  p.hull_vertex_count = ints("collision_hull_vertex_count");
  p.hull_vertices = doubles("collision_hull_vertices");
  out.joint_names = strings("collision_joint_names");
  out.joint_aliases = strings("collision_joint_aliases");
  return out;
}

}  // namespace openral_octomap_bridge

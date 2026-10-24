// SPDX-License-Identifier: Apache-2.0
// SafetyKernelLifecycleNode source.

#include "openral_safety_kernel/lifecycle_kernel.hpp"

#include "openral_safety_kernel/otel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

#include <opentelemetry/common/attribute_value.h>
#include <opentelemetry/context/runtime_context.h>
#include <opentelemetry/trace/scope.h>
#include <opentelemetry/trace/span.h>
#include <opentelemetry/trace/tracer.h>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <openral_msgs/msg/attached_collision_primitive.hpp>
#include <rclcpp/qos.hpp>

namespace openral_safety_kernel {

namespace {

// Largest per-component difference (m for translation and half-extents, unit
// rotation-matrix entries) between a re-measured grasp region and the one
// latched at handover that still counts as the same box: float noise only.
constexpr double kGraspRegionLatchNoise = 1e-6;

/// A wire quaternion the kernel may turn into a rotation: finite and unit to
/// 1e-6. `transform_from_translation_quat` reads the all-zero (unset) quaternion
/// as identity, so a box or grid posed by one would sit on the wrong axes —
/// fail-OPEN wherever that geometry exempts or clears anything. Templated on the
/// wire type (`geometry_msgs` Quaternion, reached through `openral_msgs`).
template <typename Quaternion>
bool is_unit_quaternion(const Quaternion& q) noexcept {
  const double norm2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  return std::isfinite(norm2) && std::fabs(norm2 - 1.0) <= 1e-6;
}

/// Escape a string for a JSON string literal (quotes, backslashes, control chars).
std::string json_escape(std::string_view in) {
  std::string out;
  out.reserve(in.size());
  for (const char c : in) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
        out += buf;
      } else {
        out += c;
      }
    }
  }
  return out;
}

bool has_calibration_name(const std::string& reference) noexcept {
  return reference.find_first_not_of(" \t\n\r\f\v") != std::string::npos;
}

const char* workspace_denial_reason(int code) noexcept {
  switch (code) {
  case 0:
    return "not_applicable_or_exempt";
  case 1:
    return "not_hand_link";
  case 2:
    return "hand_speed_over_limit_or_unknown";
  case 3:
    return "hand_lead_over_limit_or_unknown";
  case 4:
    return "cell_center_outside_workspace";
  case 5:
    return "cell_unexplained";
  default:
    return nullptr;
  }
}

rclcpp::QoS chunk_qos() {
  // openral slot dispatcher publishes N chunks/tick on /openral/candidate_action
  // (arm CARTESIAN_DELTA + gripper GRIPPER_POSITION + optional base
  // BODY_TWIST). KEEP_LAST=1 would coalesce back-to-back publishes in one
  // callback batch to only the last slot (arm freezes in deploy_sim while
  // the gripper keeps streaming, since gripper is the LAST slot per tick).
  // Four-slot mobile manipulation + predictive collision can backlog past
  // ten samples; 50 holds twelve complete ticks while staying bounded.
  rclcpp::QoS q(rclcpp::KeepLast(50));
  q.reliable();
  q.durability_volatile();
  return q;
}

rclcpp::QoS estop_qos() {
  rclcpp::QoS q(rclcpp::KeepLast(10));
  q.reliable();
  q.durability_volatile();
  return q;
}

rclcpp::QoS failure_qos() {
  rclcpp::QoS q(rclcpp::KeepLast(50));
  q.reliable();
  q.durability_volatile();
  return q;
}

rclcpp::QoS safety_status_qos() {
  // ADR-0096 — the LATCHED status topic. RELIABLE + TRANSIENT_LOCAL +
  // KEEP_LAST=1 is CLAUDE.md §2's "description/static" profile, deliberately
  // NOT the "safety/e-stop" profile estop_qos()/failure_qos() use above:
  // /openral/safety_status answers "what is true right now", which a
  // late-joining subscriber (a dashboard opened mid-mission, a runner
  // reconnecting after a crash) must be able to read without having
  // witnessed the transition. VOLATILE cannot deliver that by construction.
  rclcpp::QoS q(rclcpp::KeepLast(1));
  q.reliable();
  q.transient_local();
  return q;
}

// Correlation ids for transitions with no chunk behind them (activation,
// external e-stop, operator reset). Named rather than a bare temporary so the
// intent — "we genuinely do not know which skill" — is on the page.
const std::string kNoCorrelationId;  // NOLINT(cert-err58-cpp)

const char* violation_kind_field(ViolationKind k) {
  switch (k) {
  case ViolationKind::kForce:
    return "force";
  case ViolationKind::kWorkspace:
    return "workspace";
  case ViolationKind::kController:
    return "controller";
  case ViolationKind::kCollision:
    return "collision";
  }
  return "unknown";
}

std::uint8_t violation_kind_constant(ViolationKind k) {
  // Mirrors openral_msgs/FailureTrigger constants.
  switch (k) {
  case ViolationKind::kForce:
    return 1;  // KIND_FORCE
  case ViolationKind::kWorkspace:
    return 2;  // KIND_WORKSPACE
  case ViolationKind::kController:
    return 5;  // KIND_CONTROLLER
  case ViolationKind::kCollision:
    return 10;  // KIND_COLLISION
  }
  return 5;
}

/// Rigid-transform algebra for the #418 K4 union (ingest path only).
Transform rigid_compose(const Transform& a, const Transform& b) noexcept {
  Transform out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out.r[i * 3 + j] =
          a.r[i * 3] * b.r[j] + a.r[i * 3 + 1] * b.r[3 + j] + a.r[i * 3 + 2] * b.r[6 + j];
    }
  }
  out.t = Vec3{a.t.x + a.r[0] * b.t.x + a.r[1] * b.t.y + a.r[2] * b.t.z,
               a.t.y + a.r[3] * b.t.x + a.r[4] * b.t.y + a.r[5] * b.t.z,
               a.t.z + a.r[6] * b.t.x + a.r[7] * b.t.y + a.r[8] * b.t.z};
  return out;
}
Transform rigid_inverse(const Transform& x) noexcept {
  Transform out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out.r[i * 3 + j] = x.r[j * 3 + i];
    }
  }
  out.t = Vec3{-(out.r[0] * x.t.x + out.r[1] * x.t.y + out.r[2] * x.t.z),
               -(out.r[3] * x.t.x + out.r[4] * x.t.y + out.r[5] * x.t.z),
               -(out.r[6] * x.t.x + out.r[7] * x.t.y + out.r[8] * x.t.z)};
  return out;
}

/// Same shape, dimensions and pose to 1e-9: one primitive, not two (#418 K4).
bool same_primitive(const AttachedPrimitiveInput& a, const AttachedPrimitiveInput& b) noexcept {
  const auto near = [](double x, double y) { return std::abs(x - y) <= 1e-9; };
  bool same = a.kind == b.kind && near(a.radius, b.radius) && near(a.half_length, b.half_length) &&
              near(a.half_extents.x, b.half_extents.x) &&
              near(a.half_extents.y, b.half_extents.y) &&
              near(a.half_extents.z, b.half_extents.z) &&
              near(a.pose_in_object.t.x, b.pose_in_object.t.x) &&
              near(a.pose_in_object.t.y, b.pose_in_object.t.y) &&
              near(a.pose_in_object.t.z, b.pose_in_object.t.z);
  for (int k = 0; k < 9 && same; ++k) {
    same = near(a.pose_in_object.r[k], b.pose_in_object.r[k]);
  }
  return same;
}

/// Decode one wire `AttachedCollisionPrimitive` into the kernel's input record.
///
/// Shared by attached-payload ingest and the declared place target's geometry
/// (ADR-0098) so both interpret the same wire shape identically. Returns
/// false (fail-closed) on an unknown SHAPE_* tag or too few dimensions for
/// the tag given.
/// Copy a primitive's optional stage-2 refinement off the wire (#266).
///
/// Shape only — nothing here is interpreted. `ingest_attached_objects` proves
/// the refinement is inside the primitive's own box before any of it is used,
/// and silently drops what it cannot prove. A malformed refinement therefore
/// leaves `has_tight` false and the primitive is checked as the plain box, so
/// a producer publishing nonsense cannot make the kernel's broad-phase window
/// skip a cell.
///
/// It cannot prove the refinement bounds the real payload — the kernel never
/// sees a mesh — but that is not a trust this decode adds: `shape_dimensions`
/// is producer-supplied on the same message under the same trust, so a
/// producer able to lie here could already lie there. See `accept_tight`.
void decode_tight_geometry(const openral_msgs::msg::AttachedCollisionPrimitive& prim,
                           AttachedPrimitiveInput& out) {
  out.has_tight = false;
  out.hull_vertices.clear();
  if (prim.tight_dop_lo.size() != static_cast<std::size_t>(kDopAxes) ||
      prim.tight_dop_hi.size() != static_cast<std::size_t>(kDopAxes) ||
      prim.tight_hull_vertices.size() % 3 != 0) {
    return;
  }
  for (int i = 0; i < kDopAxes; ++i) {
    out.dop_lo[i] = prim.tight_dop_lo[static_cast<std::size_t>(i)];
    out.dop_hi[i] = prim.tight_dop_hi[static_cast<std::size_t>(i)];
  }
  // Bounded before a byte is reserved: `tight_hull_vertices` is an unbounded
  // `float64[]` and this runs inside the safety node's subscriber. Past the
  // budget the vertices are worthless anyway (`accept_tight` keeps the slabs
  // and runs stage 1), so decoding them would be an unbounded allocation in
  // exchange for nothing.
  const std::size_t n = prim.tight_hull_vertices.size() / 3;
  if (n > static_cast<std::size_t>(kMaxTightHullVertices)) {
    out.has_tight = true;  // the slabs above still stand; stage 1 only
    return;
  }
  out.hull_vertices.reserve(n);
  for (std::size_t v = 0; v < n; ++v) {
    out.hull_vertices.push_back(Vec3{prim.tight_hull_vertices[3 * v],
                                     prim.tight_hull_vertices[3 * v + 1],
                                     prim.tight_hull_vertices[3 * v + 2]});
  }
  out.has_tight = true;
}

bool decode_attached_primitive(const openral_msgs::msg::AttachedCollisionPrimitive& prim,
                               AttachedPrimitiveInput& out) {
  const std::size_t n_dims = prim.shape_dimensions.size();
  if (prim.shape_type == openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_SPHERE) {
    if (n_dims < 1) {
      return false;
    }
    out.kind = AttachedShapeKind::kSphere;
    out.radius = prim.shape_dimensions[0];
    out.half_length = 0.0;
  } else if (prim.shape_type == openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_CAPSULE) {
    if (n_dims < 2) {
      return false;
    }
    out.kind = AttachedShapeKind::kCapsule;
    out.radius = prim.shape_dimensions[0];
    // The wire carries the full central-segment length; the kernel capsule uses
    // the half-length convention.
    out.half_length = 0.5 * prim.shape_dimensions[1];
  } else if (prim.shape_type == openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_BOX) {
    if (n_dims < 3) {
      return false;
    }
    out.kind = AttachedShapeKind::kBox;
    out.half_extents =
        Vec3{prim.shape_dimensions[0], prim.shape_dimensions[1], prim.shape_dimensions[2]};
  } else {
    return false;
  }
  out.pose_in_object = transform_from_translation_quat(
      prim.pose_in_object.position.x, prim.pose_in_object.position.y,
      prim.pose_in_object.position.z, prim.pose_in_object.orientation.x,
      prim.pose_in_object.orientation.y, prim.pose_in_object.orientation.z,
      prim.pose_in_object.orientation.w);
  decode_tight_geometry(prim, out);
  // Issue #418 OT-10 amendment: the evidence behind the primitive. 0 and every code this kernel
  // does not know is FALLBACK, which explains nothing.
  out.source = prim.geometry_source == 1   ? PrimitiveSource::kMeasured
               : prim.geometry_source == 2 ? PrimitiveSource::kGrown
                                           : PrimitiveSource::kFallback;
  return true;
}

}  // namespace

SafetyKernelLifecycleNode::SafetyKernelLifecycleNode(const std::string& node_name,
                                                     const rclcpp::NodeOptions& options)
    : rclcpp_lifecycle::LifecycleNode(node_name, options) {
  // CLAUDE.md §1.4: declare parameters at construction; don't depend on
  // launch-file presence.
  this->declare_parameter<double>("estop_reset_cooldown_s", kDefaultEstopResetCooldownSec);
  this->declare_parameter<std::int64_t>("chunk_validation_deadline_us",
                                        kDefaultChunkValidationDeadlineUs);

  // Parameter-based envelope source (2026-05-24). The
  // Python `deploy_e2e.launch.py` unpacks `robots/<id>/robot.yaml` via
  // Pydantic, calls
  // `openral_safety.envelope_loader.kernel_params_from_envelope`, and
  // forwards each field as a ROS parameter here. There is exactly one
  // transport: ROS parameters. The flat-YAML `envelope_file:=PATH`
  // path the kernel had before this is gone.
  this->declare_parameter<std::int64_t>("n_dof", 0);
  this->declare_parameter<std::string>("robot_name", "");
  this->declare_parameter<std::string>("rskill_id", "");
  this->declare_parameter<std::string>("skill_revision", "");
  this->declare_parameter<std::vector<double>>("joint_position_min", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("joint_position_max", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("joint_velocity_max", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("joint_torque_max", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("workspace_box_min_xyz", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("workspace_box_max_xyz", std::vector<double>{});
  this->declare_parameter<double>("max_ee_speed_m_s", kPosInfinity);
  this->declare_parameter<double>("max_ee_accel_m_s2", kPosInfinity);
  this->declare_parameter<double>("max_ee_angular_speed_rad_s", kPosInfinity);
  this->declare_parameter<double>("max_force_n", kPosInfinity);
  this->declare_parameter<double>("max_torque_nm", kPosInfinity);
  this->declare_parameter<double>("contact_force_threshold_n", kPosInfinity);
  this->declare_parameter<double>("max_base_linear_speed_m_s", kPosInfinity);
  this->declare_parameter<double>("max_base_angular_speed_rad_s", kPosInfinity);
  this->declare_parameter<double>("max_cartesian_step_m", kPosInfinity);
  this->declare_parameter<double>("max_cartesian_step_rad", kPosInfinity);
  this->declare_parameter<bool>("deadman_required", false);
  this->declare_parameter<std::vector<std::string>>("gripper_ee_names", std::vector<std::string>{});
  this->declare_parameter<std::vector<double>>("gripper_command_min", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("gripper_command_max", std::vector<double>{});

  // Self-collision model. Disabled unless the launch emits a
  // populated model (openral_safety.envelope_loader.collision_params_from_*).
  // Flat parallel arrays mirror the per-joint envelope arrays above.
  this->declare_parameter<bool>("self_collision_enabled", false);
  this->declare_parameter<double>("self_collision_margin_m", 0.0);
  this->declare_parameter<std::int64_t>("collision_n_links", 0);
  this->declare_parameter<std::vector<std::int64_t>>("collision_parent",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<std::int64_t>>("collision_joint_kind",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<std::int64_t>>("collision_dof_index",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<double>>("collision_origin_xyzrpy", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("collision_axis", std::vector<double>{});
  this->declare_parameter<std::vector<std::int64_t>>("collision_capsule_link",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<double>>("collision_capsule_radius", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("collision_capsule_half_length",
                                               std::vector<double>{});
  this->declare_parameter<std::vector<double>>("collision_capsule_origin_xyzrpy",
                                               std::vector<double>{});
  // OBB primitive for blocky links (e.g. SO-ARM base; issue #84).
  this->declare_parameter<std::vector<std::int64_t>>("collision_box_link",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<double>>("collision_box_half_extents", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("collision_box_origin_xyzrpy",
                                               std::vector<double>{});
  // Tight geometry refining a subset of those OBBs for the world-voxel check
  // only (a 26-DOP, plus an exact convex hull when it fits the kernel's vertex
  // budget). CSR-shaped: `collision_box_hull` is parallel to
  // `collision_box_link`; the `collision_hull_*` arrays are indexed by it.
  // Empty means every box keeps the shipped `box_box_distance` narrow phase.
  this->declare_parameter<std::vector<std::int64_t>>("collision_box_hull",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<double>>("collision_hull_dop_lo", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("collision_hull_dop_hi", std::vector<double>{});
  this->declare_parameter<std::vector<std::int64_t>>("collision_hull_vertex_first",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<std::int64_t>>("collision_hull_vertex_count",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<double>>("collision_hull_vertices", std::vector<double>{});
  // Mimic joints (`RobotDescription.mimic_joints`, the URDF `<mimic>`): a link
  // of `collision_joint_kind` 3/4 reads the column `collision_dof_index` names
  // and moves by `multiplier * q + offset`. Per link, parallel to
  // `collision_parent`; empty = no mimic link (identity for every link).
  this->declare_parameter<std::vector<double>>("collision_mimic_multiplier", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("collision_mimic_offset", std::vector<double>{});
  this->declare_parameter<std::vector<std::int64_t>>("collision_allowed_pairs",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<std::string>>("collision_link_names",
                                                    std::vector<std::string>{});

  // Issue #373 — external joint-torque observer (collision tier). Off unless
  // the launch emits a model (openral_safety.envelope_loader
  // .collision_params_from_description, from RobotDescription.force_observer).
  // Per-link arrays are parallel to `collision_parent`; the joint arrays
  // describe the joint INTO that link. `joint_external_torque_max` is the
  // envelope's per-dof ceiling (N·m, 0 = not monitored).
  this->declare_parameter<bool>("force_observer_enabled", false);
  this->declare_parameter<double>("force_observer_gain_per_s", 50.0);
  this->declare_parameter<double>("force_observer_coulomb_velocity_rad_s", 0.05);
  this->declare_parameter<double>("force_observer_limit_margin_rad", 0.02);
  this->declare_parameter<std::vector<double>>("force_observer_gravity",
                                               std::vector<double>{0.0, 0.0, -9.81});
  this->declare_parameter<std::vector<double>>("force_observer_link_mass", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("force_observer_link_com", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("force_observer_link_inertia",
                                               std::vector<double>{});
  this->declare_parameter<std::vector<double>>("force_observer_joint_armature",
                                               std::vector<double>{});
  this->declare_parameter<std::vector<double>>("force_observer_joint_viscous",
                                               std::vector<double>{});
  this->declare_parameter<std::vector<double>>("force_observer_joint_coulomb",
                                               std::vector<double>{});
  this->declare_parameter<std::vector<double>>("force_observer_joint_offset",
                                               std::vector<double>{});
  this->declare_parameter<std::vector<double>>("joint_external_torque_max", std::vector<double>{});

  // Voxel phase — dense occupancy-grid world check (octomap path).
  this->declare_parameter<bool>("world_voxel_enabled", false);
  this->declare_parameter<double>("world_voxel_margin_m", 0.0);
  this->declare_parameter<double>("world_voxel_deadline_ms", 500.0);
  // How old the sensor data behind the grid may be at check time, from the
  // grid's `source_stamp` (the capture stamp of the newest cloud in the
  // octree). The deadline above is measured from receipt and cannot see the
  // pipeline's latency; this one can. A grid with no `source_stamp` is stale.
  // There is no "off": with the world check enabled, configure refuses a
  // budget outside (0, kMaxWorldVoxelDataAgeBudgetMs] and a deadline outside
  // (0, kMaxWorldVoxelDeadlineMs], so a node launched directly (not through
  // `openral deploy`) is held to DeployRuntime's caps too.
  this->declare_parameter<double>("world_voxel_data_age_budget_ms",
                                  kDefaultWorldVoxelDataAgeBudgetMs);
  // Sized by the ROBOT, not by taste: must cover wherever the kernel-checked
  // geometry can reach — a ball on a lattice-aligned grid (axes are the
  // map's, turn relative to base, so a base-aligned box isn't invariant).
  // panda_mobile's checked arm reaches 1016 mm from grid centre; deploy
  // launch covers 1.05 m, 85^3 = 614125 cells at sim's 25 mm resolution
  // (includes the one lattice-snap cell per axis).
  //
  // Previous cap 262144 (1.6 m at 25 mm = 64^3) was sized to the sim box, not
  // the robot — left the arm reaching 124 mm outside the published grid,
  // unchecked.
  //
  // Cost is dominated by the attached-contact baseline, not occupancy:
  // attached_contact_distance_ is attached_max_objects x 8 B/cell vs
  // occupancy's 1 B/cell — this default reserves ~40 MB, 39 MB of it that
  // baseline. Sparse storage is the lever if that ever binds.
  this->declare_parameter<std::int64_t>("world_voxel_max_cells", 614125);

  // Attached-payload phase — grasped objects carried on
  // /openral/world_state_fast (ADR-0092). Each object leaves world occupancy
  // (openral_octomap_bridge clears its own cells from /openral/world_voxels
  // off this same message; the kernel exempts nothing itself) and is
  // re-checked as collision-active robot geometry (vs world, voxels, and the
  // robot's own links except attach link + touch links). Fixed-capacity
  // caps: over-capacity, unknown-link, or malformed attachment fails closed
  // (next candidate dropped until a clean message lands). Each object owns
  // one identity/attach link/touch-link set but may carry several
  // primitives; object/primitive/touch-link caps enforce independently.
  this->declare_parameter<bool>("attached_collision_enabled", false);
  this->declare_parameter<double>("attached_collision_margin_m", 0.0);
  // Consecutive advisory refusals (#176) before the kernel treats the next
  // one as an ordinary latched stop. 0 disables the band entirely (exact
  // pre-#176 behaviour) — set that if a deployment wants no part of it.
  //
  // 3 is small on purpose: the band exists so an ARRIVED place is not an
  // operator-reset event, not to let a skill push repeatedly. A skill that
  // can't clear the band in three attempts isn't grazing the receptacle;
  // the fourth refusal latches.
  this->declare_parameter<std::int64_t>("place_advisory_max_consecutive", 3);

  // Distance-graded velocity scaling (#188). Width of the band ABOVE each
  // check's own gate margin where an accepted chunk is slowed, not passed at
  // full rate. 0 (default) disables the mechanism entirely, reproducing
  // today's behaviour bit-for-bit — the A/B five-round battery earns a
  // non-zero value, not this parameter's existence.
  //
  // scale = exp(k * (slack - proximity)): 1.0 at the band's top,
  // exp(-k * proximity) at the margin. Suggested 0.05 m / 20.0 -> 1.00 to
  // 0.37 across the band.
  this->declare_parameter<double>("collision_scale_proximity_m", 0.0);
  this->declare_parameter<double>("collision_scale_k", 20.0);
  // Floor on the scale. A band that can reach zero does not slow the robot,
  // it stops it without saying so — the deadlock the literature does not
  // settle in either direction (survey §10). The floor is what keeps "graded"
  // from becoming an unlogged stop; the latch below the margin is still the
  // only thing that stops the robot.
  this->declare_parameter<double>("collision_scale_min", 0.1);

  this->declare_parameter<double>("attached_collision_deadline_ms", 500.0);
  // Physical slack added to an ATTESTED support-contact depth (ADR-0092 D6) —
  // FK and pose noise, not voxel quantisation, which the witness predicate
  // accounts for geometrically. 1 mm is the honest physical figure; there is no
  // longer any reason to raise it to the voxel size in simulation.
  this->declare_parameter<double>("attached_contact_tolerance_m", 0.001);
  // Bounds on what a layer-2 attestation may claim. A witness beyond these
  // fails the whole attachment message closed.
  this->declare_parameter<double>("support_witness_max_patch_radius_m", 0.5);
  this->declare_parameter<double>("support_witness_max_penetration_m", 0.01);
  this->declare_parameter<std::int64_t>("attached_max_objects", 8);
  this->declare_parameter<std::int64_t>("attached_max_primitives", 16);
  this->declare_parameter<std::int64_t>("attached_max_touch_links", 32);
  // Grasp-target exemption (ADR-0115 draft, hazard HZ-0115; Safety-WG review
  // pending). Default OFF: nothing is exempt and the world-voxel check is
  // unchanged. When on, `grasp_contact_links` is the launch-derived allowlist
  // (the manifest's `role: gripper` joints' child links); a declaration may
  // exempt only links on it, and an unknown name fails configure.
  this->declare_parameter<bool>("grasp_allowance_enabled", false);
  this->declare_parameter<std::vector<std::string>>("grasp_contact_links",
                                                    std::vector<std::string>{});
  // How old a producer-measured region (`region.stamp_ns`) may be and still
  // exempt anything — grasp target and place target respectively. Regions are
  // perceived from the voxel map, so the declaration's own `timeout_s` (up to
  // 120 s) is no bound on the measurement. 0 = derive 2 x world_voxel_deadline;
  // configure refuses a resolved value outside (0, kMaxRegionMeasurementAgeS]
  // while the matching allowance is enabled.
  this->declare_parameter<double>("grasp_region_max_age_s", 0.0);
  // Issue #418 K2 (hazard OT-6; Safety-WG review pending): the SOURCE of the grasp
  // exemption. false (default) = the producer-measured region above, unchanged;
  // true = the tracked-object table replaces it (finger links vs the unique nearest
  // eligible tracked object per hand, cover re-checked here). Not a disable: either
  // way only the declared fingers can ever be exempt. Set by the launch for
  // `object_tracking: on` only; configure refuses it without
  // `grasp_allowance_enabled` or off `hal_mode: sim`.
  this->declare_parameter<bool>("tracked_grasp_exemption_enabled", false);
  // Issue #418 K2 (hazard OT-6 amendment; Safety-WG review pending): fixed jaws that
  // are body links (the manifest's `fixed_jaw_link`, the SO-101's `gripper`). Read
  // ONLY by the tracked grasp exemption, which may exempt them against the selected
  // tracked object's box like the hand's fingers; `grasp_contact_links` (the legacy
  // region and the declaration check) is unchanged. Configure refuses a name that is
  // no collision link, a hand of these alone, or any name while K2 is off.
  this->declare_parameter<std::vector<std::string>>("tracked_grasp_extra_contact_links",
                                                    std::vector<std::string>{});
  // Issue #418 K7 (hazard OT-11; Safety-WG review pending): explained cells for
  // workspace contact. false (default) = the #349 workspace rule, unchanged; true =
  // a (hand link, cell) pair the workspace would tolerate is tolerated only on a
  // cell an explaining tracked object's box, a declared static collider or the
  // measured support under such an object explains — a NEW conjunct, so it only
  // ever adds stops. Set by the launch for `object_tracking: on` only; configure
  // refuses it without `workspace_contact_enabled` (itself sim-only).
  this->declare_parameter<bool>("workspace_explained_cells_enabled", false);
  this->declare_parameter<bool>("payload_own_cells_enabled", false);
  this->declare_parameter<bool>("pinch_payload_own_cells_enabled", false);
  // The declared static geometry as base-frame surface points, one `x y z` per
  // line — the SAME file the octomap bridge marks into every grid
  // (`static_points_file`). K7 reads it to know which cells are a declared static
  // collider. "" = none declared (no cell is explained as static).
  this->declare_parameter<std::string>("static_points_file", "");
  this->declare_parameter<double>("place_region_max_age_s", 0.0);
  // Manipulation-workspace contact (issue #349, draft; Safety-WG review
  // pending). Default OFF. When on: a hand link (`workspace_contact_links`, the
  // launch-derived hand links — gripper leaders, mimic followers and the link
  // carrying them) inside the live grasp declaration's producer-confirmed
  // `workspace_region` does not trip the world-voxel check while the hand's
  // speed is at most `workspace_contact_max_speed_mps` AND its lead over the
  // measured pose is at most `workspace_contact_max_lead_m`. The speed is the
  // MEASURED hand speed (/joint_states velocity over one
  // `workspace_contact_control_period_s`); without a fresh measured velocity,
  // the commanded speed (row spacing, row 0 from the previously accepted
  // command while fresh) is the fallback.
  // No default the kernel would silently use: enabled with a speed, lead or
  // period that is not finite and > 0 (lead also <= kMaxWorkspaceContactLeadM)
  // refuses configure, as does enabled without `grasp_allowance_enabled`.
  this->declare_parameter<bool>("workspace_contact_enabled", false);
  // The HAL mode the graph runs (`sim` | `real`). The workspace rule is sim-only until
  // Safety-WG review, and the kernel enforces that itself rather than trusting every
  // launch to (#349 review). Defaults to real: an unset mode never arms a sim-only rule.
  this->declare_parameter<std::string>("hal_mode", "real");
  // ADR-0121: the exemption provider (a pluginlib class deriving
  // `openral_safety_kernel::ExemptionProvider`). Empty = none, today's verdicts. Sim only.
  // `exemption_provider_links` is the kernel's own relaxable-link mask (the manifest's hand
  // links, derived by the launch); the provider is never asked about any other link.
  this->declare_parameter<std::string>("exemption_provider", "");
  this->declare_parameter<std::vector<std::string>>("exemption_provider_links",
                                                    std::vector<std::string>{});
  this->declare_parameter<std::vector<std::string>>("workspace_contact_links",
                                                    std::vector<std::string>{});
  this->declare_parameter<double>("workspace_contact_max_speed_mps", 0.0);
  this->declare_parameter<double>("workspace_contact_max_lead_m", 0.0);
  // The gripper (jaw) joints, from the manifest's `role: gripper` joints: the lead
  // is taken with them at their measured value, so it bounds the ARM's push and
  // not a closing jaw's remaining stroke (a grasp squeeze is bounded by the
  // gripper's own effort limit). Empty = every joint counts (the stricter reading).
  this->declare_parameter<std::vector<std::string>>("workspace_contact_jaw_joints",
                                                    std::vector<std::string>{});
  // Issue #418 K5, the kernel's jaw-open release of a TRACKED_OBJECT payload (NOT
  // monotone, hazard OT-7; Safety-WG review pending). Empty jaw list = off, today's
  // semantics. `open_sign` (+1 / -1 per jaw) says which way the joint opens; `gap`
  // is in the joint's own units past the hold measured at attach.
  this->declare_parameter<std::vector<std::string>>("tracked_release_jaw_joints",
                                                    std::vector<std::string>{});
  this->declare_parameter<std::vector<double>>("tracked_release_jaw_open_sign",
                                               std::vector<double>{});
  this->declare_parameter<double>("tracked_release_gap", 0.0);
  this->declare_parameter<double>("tracked_release_dwell_s", 0.0);
  this->declare_parameter<double>("tracked_release_clear_m", 0.04);
  this->declare_parameter<double>("tracked_release_timeout_s", 3.0);
  // #349 workspace brake: with the workspace rule on, a position chunk whose only
  // reason to stop is the measured hand speed holds the measured pose instead.
  this->declare_parameter<bool>("workspace_contact_brake_enabled", false);
  // ... and, up to this many consecutive chunks, a command the relaxed check still
  // refuses while the measured pose passes is held too (MoveIt Servo's halt: the
  // command is never executed, the arm stays where it is clear); past it, today's
  // latched stop. 0 = off.
  this->declare_parameter<std::int64_t>("workspace_contact_hold_max_consecutive", 0);
  this->declare_parameter<double>("workspace_contact_control_period_s", 0.0);

  // Measured joint-state seed for non-position collision checks.
  // `collision_joint_names` is the actuated joint order (length n_dof) the
  // launch forwards from the robot manifest; it maps /joint_states names to the
  // action's dof index. `collision_seed_dt_s` is the velocity-integration step
  // (the control period); 0 disables the predictive look-ahead but the reactive
  // measured-config check still runs. `collision_state_deadline_ms` bounds how
  // stale the measured seed may be before a seed-requiring chunk is rejected.
  this->declare_parameter<std::vector<std::string>>("collision_joint_names",
                                                    std::vector<std::string>{});
  this->declare_parameter<double>("collision_seed_dt_s", 0.0);
  this->declare_parameter<double>("collision_state_deadline_ms", 200.0);
  // Dof indices of the planar mobile-base joints (description
  // base_joints). They are zeroed before the base-relative geometric FK so the
  // arm is placed in the base_link frame the world/voxel grid lives in;
  // otherwise FK applies the base's world pose and the arm sits metres outside
  // the local map. Empty for fixed-base arms (no-op).
  this->declare_parameter<std::vector<std::int64_t>>("collision_base_dofs",
                                                     std::vector<std::int64_t>{});

  // Phase 3 — predictive Cartesian look-ahead. EE collision-link index lets
  // the kernel build the arm Jacobian and reconstruct where a CARTESIAN_DELTA
  // chunk's EE deltas drive the arm; <0 (default) disables predictive
  // Cartesian (reactive measured-config check only). Launch derives the
  // index from the robot's EE frame; lambda damps the DLS solve near
  // singularities; margin_growth inflates the collision margin per
  // additional look-ahead step to bound accumulated linearization/DLS
  // residual (first predicted step uses the configured margin); max_steps
  // caps the look-ahead (0 = every row, last step always included).
  this->declare_parameter<std::int64_t>("collision_ee_link_index", -1);
  this->declare_parameter<double>("collision_predict_lambda", 0.05);
  this->declare_parameter<double>("collision_predict_margin_growth_m", 0.01);
  this->declare_parameter<std::int64_t>("collision_predict_max_steps", 0);
}

SafetyKernelLifecycleNode::CallbackReturn
SafetyKernelLifecycleNode::on_configure(const rclcpp_lifecycle::State& /*state*/) {
  estop_reset_cooldown_s_ = this->get_parameter("estop_reset_cooldown_s").as_double();

  // Stand up the OTel TracerProvider before we start handling chunks so
  // the very first ``safety.check`` span lands on the collector. The
  // initializer is idempotent across lifecycle restarts — same-process
  // re-configure returns false and reuses the existing provider.
  otel::initialize_tracing();

  // Load envelope from ROS parameters. The Python
  // `deploy_e2e.launch.py` populates each field from
  // `robots/<id>/robot.yaml` via Pydantic +
  // `openral_safety.envelope_loader.kernel_params_from_envelope`.
  // CLAUDE.md §1.4 — explicit failure, no fallback: when `n_dof=0` the
  // loader returns kUnconfigured and we refuse to leave UNCONFIGURED
  // so a misboot never lets unvalidated chunks reach the HAL.
  envelope_loaded_ = false;
  std::string err;
  const EnvelopeLoadStatus rc = load_envelope_from_ros_parameters(*this, envelope_, err);
  if (rc != EnvelopeLoadStatus::kOk) {
    RCLCPP_ERROR(this->get_logger(), "envelope load failed (%d): %s", static_cast<int>(rc),
                 err.c_str());
    return CallbackReturn::FAILURE;
  }
  envelope_loaded_ = true;
  RCLCPP_INFO(this->get_logger(), "envelope loaded from ROS params: robot=%s rskill=%s n_dof=%zu",
              envelope_.robot_name.c_str(), envelope_.rskill_id.c_str(), envelope_.n_dof);

  // Load the optional self-collision model. A malformed model when
  // the feature is enabled is a configuration error: refuse to leave
  // UNCONFIGURED rather than run with a broken safety check (§1.4 fail-closed).
  std::string coll_err;
  if (!load_collision_model(coll_err)) {
    RCLCPP_ERROR(this->get_logger(), "self-collision model load failed: %s", coll_err.c_str());
    envelope_loaded_ = false;
    return CallbackReturn::FAILURE;
  }
  if (!load_force_observer(coll_err)) {
    RCLCPP_ERROR(this->get_logger(), "force observer load failed: %s", coll_err.c_str());
    envelope_loaded_ = false;
    return CallbackReturn::FAILURE;
  }
  // Grasp-target exemption allowlist (ADR-0115 draft). Resolved against the
  // loaded collision link names; a name the model does not know is a
  // misconfiguration of a safety surface and refuses configure rather than
  // silently exempting nothing (or, worse, the wrong index).
  grasp_allowance_enabled_ = this->get_parameter("grasp_allowance_enabled").as_bool();
  grasp_allowlist_.reset();
  if (grasp_allowance_enabled_) {
    const auto names = this->get_parameter("grasp_contact_links").as_string_array();
    if (names.empty()) {
      RCLCPP_ERROR(this->get_logger(),
                   "grasp_allowance_enabled with an empty grasp_contact_links allowlist");
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    }
    for (const auto& name : names) {
      const auto it = std::find(collision_link_names_.begin(), collision_link_names_.end(), name);
      const auto index = static_cast<std::size_t>(it - collision_link_names_.begin());
      if (it == collision_link_names_.end() || index >= kMaxGraspMaskLinks) {
        RCLCPP_ERROR(this->get_logger(),
                     "grasp_contact_links names '%s', which is not a collision link of this "
                     "model (or is past the %zu-link mask)",
                     name.c_str(), kMaxGraspMaskLinks);
        envelope_loaded_ = false;
        return CallbackReturn::FAILURE;
      }
      grasp_allowlist_.set(index);
    }
    RCLCPP_WARN(this->get_logger(),
                "safety.grasp_allowance enabled (ADR-0115 draft, Safety-WG review pending): "
                "%zu allowlisted contact link(s); a live grasp declaration exempts them from "
                "the world-voxel check inside its measured region",
                grasp_allowlist_.count());
  }
  // Issue #418 K2. Every refusal fails configure: an exemption source whose hands
  // or scope the kernel cannot state is not run.
  tracked_grasp_enabled_ = this->get_parameter("tracked_grasp_exemption_enabled").as_bool();
  tracked_grasp_ = TrackedGraspTable{};
  voxel_grid_.tracked_grasp = nullptr;
  const auto extra_names =
      this->get_parameter("tracked_grasp_extra_contact_links").as_string_array();
  if (!extra_names.empty() && !tracked_grasp_enabled_) {
    RCLCPP_ERROR(this->get_logger(),
                 "tracked_grasp_extra_contact_links is set but tracked_grasp_exemption_enabled "
                 "is off: refusing a K2-only parameter with no K2 to read it");
    envelope_loaded_ = false;
    return CallbackReturn::FAILURE;
  }
  if (tracked_grasp_enabled_) {
    const char* why = nullptr;
    std::bitset<kMaxGraspMaskLinks> extra;
    for (const auto& name : extra_names) {
      const auto it = std::find(collision_link_names_.begin(), collision_link_names_.end(), name);
      const auto index = static_cast<std::size_t>(it - collision_link_names_.begin());
      if (it == collision_link_names_.end() || index >= kMaxGraspMaskLinks) {
        why = "a tracked_grasp_extra_contact_links name is not a collision link";
        break;
      }
      extra.set(index);
    }
    if (why == nullptr && !grasp_allowance_enabled_) {
      why = "it requires grasp_allowance_enabled (the allowlist and the declaration)";
    } else if (why == nullptr && this->get_parameter("hal_mode").as_string() != "sim") {
      why = "sim-only until Safety-WG review (issue #418, hazard OT-6)";
    } else if (why == nullptr &&
               !build_grasp_hands(collision_model_, grasp_allowlist_, tracked_grasp_, extra)) {
      why = "the grasp_contact_links (with the extra fixed jaws) do not group into 1..4 hands "
            "each holding a jaw";
    }
    if (why != nullptr) {
      RCLCPP_ERROR(this->get_logger(), "tracked_grasp_exemption_enabled refused: %s, hal_mode=%s",
                   why, this->get_parameter("hal_mode").as_string().c_str());
      tracked_grasp_enabled_ = false;
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    }
    tracked_grasp_.valid = true;
    RCLCPP_WARN(this->get_logger(),
                "safety.tracked_grasp_exemption enabled (issue #418 K2, Safety-WG review "
                "pending): hands=%zu extra_fixed_jaws=%zu; the tracked-object table replaces "
                "the producer region as the grasp exemption source",
                tracked_grasp_.n_hands, tracked_grasp_.extra_links.count());
  }
  // Issue #418 K5. Every refusal fails configure: a release rule whose jaw,
  // direction or bounds it cannot state is not run.
  release_jaw_cols_.clear();
  release_jaw_sign_.clear();
  {
    const auto jaws = this->get_parameter("tracked_release_jaw_joints").as_string_array();
    const auto signs = this->get_parameter("tracked_release_jaw_open_sign").as_double_array();
    release_gap_ = this->get_parameter("tracked_release_gap").as_double();
    release_dwell_s_ = this->get_parameter("tracked_release_dwell_s").as_double();
    release_clear_m_ = this->get_parameter("tracked_release_clear_m").as_double();
    release_timeout_s_ = this->get_parameter("tracked_release_timeout_s").as_double();
    const auto names = this->get_parameter("collision_joint_names").as_string_array();
    std::string why;
    if (!jaws.empty()) {
      if (this->get_parameter("hal_mode").as_string() != "sim") {
        // Mirrors K2: K5 is not monotone (hazard OT-7), so the kernel keeps it off
        // real hardware itself, not only the launch's sim-only rule for bind / on.
        why = "sim-only until Safety-WG review (issue #418, hazard OT-7), hal_mode=" +
              this->get_parameter("hal_mode").as_string();
      } else if (signs.size() != jaws.size()) {
        why = "tracked_release_jaw_open_sign must give one +1/-1 per jaw joint";
      } else if (!std::isfinite(release_gap_) || release_gap_ <= 0.0) {
        why = "tracked_release_gap must be finite and > 0";
      } else if (!std::isfinite(release_dwell_s_) || release_dwell_s_ <= 0.0) {
        why = "tracked_release_dwell_s must be finite and > 0";
      } else if (!std::isfinite(release_clear_m_) || release_clear_m_ <= 0.0 ||
                 release_clear_m_ > 0.1) {
        why = "tracked_release_clear_m must be in (0, 0.1]";  // the HAL's MAX_RELEASE_CLEAR_M
      } else if (!std::isfinite(release_timeout_s_) || release_timeout_s_ <= 0.0 ||
                 release_timeout_s_ > 10.0) {
        why = "tracked_release_timeout_s must be in (0, 10]";
      }
      for (std::size_t k = 0; k < jaws.size() && why.empty(); ++k) {
        const auto it = std::find(names.begin(), names.end(), jaws[k]);
        const auto d = static_cast<std::size_t>(it - names.begin());
        if (it == names.end() || d >= envelope_.n_dof) {
          why = "tracked_release_jaw_joints names '" + jaws[k] + "', not a joint of this model";
        } else if (signs[k] != 1.0 && signs[k] != -1.0) {
          why = "tracked_release_jaw_open_sign entries must be +1 or -1";
        } else {
          release_jaw_cols_.push_back(static_cast<int>(d));
          release_jaw_sign_.push_back(signs[k]);
        }
      }
    }
    if (!why.empty()) {
      RCLCPP_ERROR(this->get_logger(), "tracked_release refused: %s", why.c_str());
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    }
    if (!release_jaw_cols_.empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "safety.tracked_release enabled (issue #418 K5, NOT monotone, hazard OT-7, "
                  "Safety-WG review pending): a TRACKED_OBJECT payload whose measured jaw opens "
                  "> %g past its hold for %g s is frozen in base; its hand and touch links stay "
                  "exempt against it until %g m clear by FK or %g s",
                  release_gap_, release_dwell_s_, release_clear_m_, release_timeout_s_);
    }
  }
  // Region measurement age bounds (kernel-region-age). Resolved after the
  // collision model so `attached_collision_enabled_` (the place allowance's
  // switch) and the voxel deadline are both known. Refused, never clamped.
  {
    const double derived_s = 2.0 * world_voxel_deadline_s_;
    const auto resolve = [derived_s](double v) { return v == 0.0 ? derived_s : v; };
    grasp_region_max_age_s_ = resolve(this->get_parameter("grasp_region_max_age_s").as_double());
    place_region_max_age_s_ = resolve(this->get_parameter("place_region_max_age_s").as_double());
    const auto in_range = [](double v) {
      return std::isfinite(v) && v > 0.0 && v <= kMaxRegionMeasurementAgeS;
    };
    const auto refused = [&](bool enabled, double value, const char* name) {
      if (!enabled || in_range(value)) {
        return false;
      }
      RCLCPP_ERROR(this->get_logger(),
                   "%s resolves to %g s, outside (0, %g]: a region measured that long ago "
                   "would still exempt links from the world-voxel check",
                   name, value, kMaxRegionMeasurementAgeS);
      return true;
    };
    if (refused(grasp_allowance_enabled_, grasp_region_max_age_s_, "grasp_region_max_age_s") ||
        refused(attached_collision_enabled_, place_region_max_age_s_, "place_region_max_age_s")) {
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    }
    RCLCPP_INFO(this->get_logger(), "safety.region_max_age grasp_s=%g place_s=%g",
                grasp_region_max_age_s_, place_region_max_age_s_);
  }
  // Manipulation-workspace contact (#349). Every refusal is a misconfiguration
  // of a safety surface: configure fails rather than running with a rule whose
  // bound or scope it cannot state.
  workspace_contact_enabled_ = this->get_parameter("workspace_contact_enabled").as_bool();
  workspace_brake_enabled_ = this->get_parameter("workspace_contact_brake_enabled").as_bool();
  workspace_brakes_ = 0;
  workspace_command_holds_.clear();
  {
    const std::int64_t cap = this->get_parameter("workspace_contact_hold_max_consecutive").as_int();
    if (cap < 0 || (cap > 0 && !workspace_brake_enabled_)) {
      RCLCPP_ERROR(this->get_logger(),
                   "workspace_contact_hold_max_consecutive refused: must be >= 0 and needs "
                   "workspace_contact_brake_enabled, got %lld",
                   static_cast<long long>(cap));
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    }
    workspace_hold_max_consecutive_ = static_cast<std::uint64_t>(cap);
  }
  if (workspace_brake_enabled_ && !workspace_contact_enabled_) {
    RCLCPP_ERROR(this->get_logger(),
                 "workspace_contact_brake_enabled refused: it requires workspace_contact_enabled");
    envelope_loaded_ = false;
    return CallbackReturn::FAILURE;
  }
  workspace_links_.reset();
  workspace_link_index_.clear();
  workspace_region_ = WorkspaceContactRegion{};
  voxel_grid_.workspace_region = WorkspaceContactRegion{};
  if (workspace_contact_enabled_ && this->get_parameter("hal_mode").as_string() != "sim") {
    RCLCPP_ERROR(this->get_logger(),
                 "workspace_contact_enabled refused: sim-only until Safety-WG review "
                 "(docs/reference/manipulation-workspace-contact.md), hal_mode=%s",
                 this->get_parameter("hal_mode").as_string().c_str());
    envelope_loaded_ = false;
    return CallbackReturn::FAILURE;
  }
  if (workspace_contact_enabled_) {
    const auto refuse = [this](const std::string& why) {
      RCLCPP_ERROR(this->get_logger(), "workspace_contact_enabled refused: %s", why.c_str());
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    };
    if (!grasp_allowance_enabled_) {
      return refuse("it requires grasp_allowance_enabled (the workspace rides the grasp "
                    "declaration)");
    }
    workspace_max_speed_mps_ = this->get_parameter("workspace_contact_max_speed_mps").as_double();
    workspace_control_period_s_ =
        this->get_parameter("workspace_contact_control_period_s").as_double();
    if (!std::isfinite(workspace_max_speed_mps_) || workspace_max_speed_mps_ <= 0.0) {
      return refuse("workspace_contact_max_speed_mps must be finite and > 0, got " +
                    std::to_string(workspace_max_speed_mps_));
    }
    if (!std::isfinite(workspace_control_period_s_) || workspace_control_period_s_ <= 0.0) {
      return refuse("workspace_contact_control_period_s must be finite and > 0, got " +
                    std::to_string(workspace_control_period_s_));
    }
    workspace_max_lead_m_ = this->get_parameter("workspace_contact_max_lead_m").as_double();
    if (!std::isfinite(workspace_max_lead_m_) || workspace_max_lead_m_ <= 0.0 ||
        workspace_max_lead_m_ > kMaxWorkspaceContactLeadM) {
      return refuse("workspace_contact_max_lead_m must be finite, > 0 and <= " +
                    std::to_string(kMaxWorkspaceContactLeadM) + ", got " +
                    std::to_string(workspace_max_lead_m_));
    }
    const auto names = this->get_parameter("workspace_contact_links").as_string_array();
    if (names.empty()) {
      return refuse("workspace_contact_links is empty");
    }
    std::string listed;
    for (const auto& name : names) {
      const auto it = std::find(collision_link_names_.begin(), collision_link_names_.end(), name);
      const auto index = static_cast<std::size_t>(it - collision_link_names_.begin());
      if (it == collision_link_names_.end() || index >= kMaxGraspMaskLinks ||
          index >= collision_model_.n_links) {
        return refuse("workspace_contact_links names '" + name +
                      "', which is not a collision link of this model");
      }
      if (!workspace_links_[index]) {
        workspace_links_.set(index);
        workspace_link_index_.push_back(static_cast<int>(index));
        listed += (listed.empty() ? "" : ",") + name;
      }
    }
    const std::size_t n_links = collision_model_.n_links;
    workspace_link_reach_.assign(n_links, 0.0);
    for (const int l : workspace_link_index_) {
      workspace_link_reach_[static_cast<std::size_t>(l)] = link_geometry_reach(collision_model_, l);
    }
    workspace_prev_plain_.assign(n_links, Transform{});
    workspace_prev_partner_.assign(n_links, Transform{});
    workspace_creep_prev_.assign(n_links, Transform{});
    workspace_meas_pose_.assign(n_links, Transform{});
    workspace_last_warn_.assign(n_links, std::chrono::steady_clock::time_point{});
    q_workspace_.assign(envelope_.n_dof, 0.0);
    q_lead_.assign(envelope_.n_dof, 0.0);
    workspace_jaw_cols_.clear();
    // Resolved against the parameter: `collision_joint_names_` is read further down.
    const auto joint_names = this->get_parameter("collision_joint_names").as_string_array();
    const auto jaw_joints = this->get_parameter("workspace_contact_jaw_joints").as_string_array();
    for (const auto& name : jaw_joints) {
      const auto it = std::find(joint_names.begin(), joint_names.end(), name);
      const auto d = static_cast<std::size_t>(it - joint_names.begin());
      if (it == joint_names.end() || d >= envelope_.n_dof) {
        return refuse("workspace_contact_jaw_joints names '" + name +
                      "', which is not a joint of this model");
      }
      workspace_jaw_cols_.push_back(static_cast<int>(d));
    }
    workspace_cmd_q_.assign(envelope_.n_dof, 0.0);
    workspace_cmd_due_ns_.assign(envelope_.n_dof, 0);
    workspace_cmd_set_.assign(envelope_.n_dof, 0);
    workspace_command_holds_.assign(envelope_.n_dof, 0);
    clear_workspace_commands(ResetReason::kConfigure);
    RCLCPP_WARN(this->get_logger(),
                "safety.workspace_contact enabled (issue #349 draft, Safety-WG review pending): "
                "hand links [%s] inside a live grasp declaration's map-confirmed workspace do "
                "not trip the world-voxel check while BOTH the hand's speed is <= %g m/s "
                "(the larger of the /joint_states velocity over one %g s period and the "
                "commanded row-to-row speed, row 0 from the previously accepted command while "
                "it is fresh, %g periods, else from the measured state) AND its lead over the "
                "measured pose is <= %g m (the push bound; no per-row allowance while the "
                "measured hand touches); arm links, contact outside the "
                "workspace and a fast or far-leading hand stop as before",
                listed.c_str(), workspace_max_speed_mps_, workspace_control_period_s_,
                kWorkspaceCommandFreshPeriods, workspace_max_lead_m_);
  }
  // Issue #418 K7. Every refusal fails configure: a narrowing whose explanation
  // sources the kernel cannot state is not run.
  workspace_explained_enabled_ = this->get_parameter("workspace_explained_cells_enabled").as_bool();
  workspace_explained_ = WorkspaceExplanation{};
  support_cells_marked_ = 0;
  support_cells_dirty_ = true;
  voxel_grid_.workspace_explained = nullptr;
  static_points_.clear();
  static_cells_.clear();
  static_cells_marked_ = 0;
  static_cells_for_ = VoxelGrid{};
  static_cells_for_valid_ = false;
  static_cells_lattice_ok_ = false;
  workspace_explained_logged_n_ = static_cast<std::size_t>(-1);
  if (workspace_explained_enabled_) {
    const auto refuse = [this](const std::string& why) {
      RCLCPP_ERROR(this->get_logger(), "workspace_explained_cells_enabled refused: %s",
                   why.c_str());
      workspace_explained_enabled_ = false;
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    };
    if (!workspace_contact_enabled_) {
      return refuse("it narrows workspace contact and requires workspace_contact_enabled");
    }
    const std::string points_file = this->get_parameter("static_points_file").as_string();
    if (!points_file.empty()) {
      std::string error;
      if (!load_static_points(points_file, static_points_, error)) {
        return refuse("static_points_file: " + error);
      }
    }
    static_cells_.assign(world_voxel_max_cells_, 0);
    support_cells_.assign(world_voxel_max_cells_, 0);
    support_queue_.assign(world_voxel_max_cells_, 0);
    support_cells_marked_ = 0;
    support_cells_dirty_ = true;
    RCLCPP_WARN(this->get_logger(),
                "safety.workspace_explained_cells enabled (issue #418 K7, Safety-WG review "
                "pending): workspace contact is tolerated only on a cell a tracked object's "
                "box (confirmed / occluded, fresh, cover-checked), a declared static collider "
                "(%zu points from '%s') or the measured support layer under such an object "
                "(within %g m of it, or connected to that through occupied in-layer cells "
                "inside the workspace) explains; every other cell in the workspace trips as "
                "outside it",
                static_points_.size(), points_file.c_str(), kExplainedSupportReachM);
  }
  // Issue #418 (hazard OT-10 amendment): a bound payload's measured cells cannot stop it. Every
  // refusal fails configure: an exemption whose scope the kernel cannot state is not run.
  payload_own_enabled_ = this->get_parameter("payload_own_cells_enabled").as_bool();
  payload_own_ = PayloadOwnCells{};
  voxel_grid_.payload_own = nullptr;
  payload_own_logged_n_ = static_cast<std::size_t>(-1);
  payload_own_disarm_logged_ = nullptr;
  if (payload_own_enabled_) {
    const auto refuse = [this](const std::string& why) {
      RCLCPP_ERROR(this->get_logger(), "payload_own_cells_enabled refused: %s, hal_mode=%s",
                   why.c_str(), this->get_parameter("hal_mode").as_string().c_str());
      payload_own_enabled_ = false;
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    };
    if (!tracked_grasp_enabled_) {
      return refuse("it requires tracked_grasp_exemption_enabled (object_tracking on)");
    }
    if (!attached_collision_enabled_) {
      return refuse(
          "it exempts the payload-vs-voxel check and requires attached_collision_enabled");
    }
    if (this->get_parameter("hal_mode").as_string() != "sim") {
      return refuse("sim-only until Safety-WG review (issue #418, hazard OT-10)");
    }
    if (!workspace_explained_enabled_) {
      // K7 loads the declared static colliders; without K7 this exemption needs them too.
      const std::string points_file = this->get_parameter("static_points_file").as_string();
      if (!points_file.empty()) {
        std::string error;
        if (!load_static_points(points_file, static_points_, error)) {
          return refuse("static_points_file: " + error);
        }
      }
      static_cells_.assign(world_voxel_max_cells_, 0);
    }
    RCLCPP_WARN(this->get_logger(),
                "safety.payload_own_cells enabled (issue #418 OT-10 amendment, Safety-WG review "
                "pending): a bound payload's measured / grown cells (not its pinch-volume "
                "fallback) cannot stop that payload when they stand above the track's floor, are "
                "no declared static cell (%zu points) and meet no other settled track's box; "
                "payload vs robot links and every hand check are unchanged",
                static_points_.size());
  }
  pinch_own_enabled_ = this->get_parameter("pinch_payload_own_cells_enabled").as_bool();
  pinch_floor_ = {};
  if (pinch_own_enabled_) {
    if (!payload_own_enabled_) {
      RCLCPP_ERROR(this->get_logger(),
                   "pinch_payload_own_cells_enabled refused: it widens payload_own_cells_enabled "
                   "and requires it");
      pinch_own_enabled_ = false;
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    }
    RCLCPP_WARN(this->get_logger(),
                "safety.pinch_payload_own_cells enabled (issue #418 OT-10 amendment, untracked "
                "holds, Safety-WG review pending): an untracked_grasp hold's pinch-volume cells "
                "cannot stop that payload above the pinch box's lowest point at the bind, unless "
                "static or in another settled track's box");
  }
  if (self_collision_enabled_) {
    RCLCPP_INFO(this->get_logger(), "self-collision check enabled: %zu links, margin=%g m",
                collision_model_.n_links, self_collision_margin_m_);
    // Disclosure, never a decision (§1.4): a reader of the log must be able to
    // tell which links the world-voxel check tightened and how far each one
    // got, without inferring it from the manifest.
    if (!collision_model_.box_hull.empty()) {
      std::size_t staged = 0;
      std::size_t exact = 0;
      for (std::size_t b = 0; b < collision_model_.box_hull.size(); ++b) {
        const int h = collision_model_.box_hull[b];
        if (h < 0) {
          continue;
        }
        ++staged;
        if (collision_model_.hulls[static_cast<std::size_t>(h)].vertex_count > 0) {
          ++exact;
        }
      }
      RCLCPP_INFO(this->get_logger(),
                  "world-voxel tight geometry: %zu of %zu boxed links staged (26-DOP), "
                  "%zu of those also exact (convex hull, <= %d vertices); "
                  "the rest keep box_box_distance",
                  staged, collision_model_.box_hull.size(), exact, kMaxTightHullVertices);
    }
  }

  if (!collision_model_.mimic_multiplier.empty()) {
    std::size_t n_mimic = 0;
    for (const JointKind kind : collision_model_.joint_kind) {
      if (is_mimic(kind)) {
        ++n_mimic;
      }
    }
    RCLCPP_INFO(this->get_logger(),
                "collision model: %zu mimic link(s) posed from another joint's column "
                "(multiplier * q + offset), rigid, checked at the commanded value",
                n_mimic);
  }

  // Set up the measured joint-state seed used to reconstruct
  // non-position chunks (Phase 1) for the velocity check (Phase 2). Sized to
  // n_dof; the name→dof map lets /joint_states (named) fill q_meas_ in the
  // action's dof order. `collision_fk_dofs_` is the set of dof indices FK
  // actually consumes (links with a movable joint) — the freshness gate
  // requires every one of them to have been observed before a velocity chunk is
  // checked, so a missing joint feed fails closed instead of FK-ing a zero pose.
  collision_joint_names_ = this->get_parameter("collision_joint_names").as_string_array();
  collision_seed_dt_s_ = this->get_parameter("collision_seed_dt_s").as_double();
  collision_state_deadline_s_ =
      this->get_parameter("collision_state_deadline_ms").as_double() / 1000.0;
  const std::size_t ndof = envelope_.n_dof;
  q_meas_.assign(ndof, 0.0);
  q_meas_seen_.assign(ndof, false);
  dq_meas_.assign(ndof, 0.0);
  dq_meas_seen_.assign(ndof, 0);
  dq_meas_stamp_ns_.assign(ndof, 0);
  q_meas_src_ns_.assign(ndof, 0);
  q_check_.assign(ndof, 0.0);
  q_fk_.assign(ndof, 0.0);
  ws_trip_q_.assign(ndof, 0.0);
  slot_commanded_.assign(ndof, 0);
  tick_target_.assign(ndof, 0.0);
  tick_target_set_.assign(ndof, 0);
  tick_target_session_ = 0;
  tick_target_tick_ = 0;
  q_meas_received_ = false;
  collision_base_dofs_.clear();
  // NB: bind the parameter's array to a NAMED local first. Iterating directly
  // over `get_parameter(...).as_integer_array()` dangles — `as_integer_array()`
  // returns a reference into the temporary `rclcpp::Parameter`, which the
  // range-based for does NOT lifetime-extend, so the loop would read freed
  // stack memory (ASAN: stack-use-after-scope). This silently left the
  // mobile-base dofs un-zeroed → base-relative FK broken → world collisions on
  // a mobile base were never caught.
  const std::vector<std::int64_t> base_dofs =
      this->get_parameter("collision_base_dofs").as_integer_array();
  for (const std::int64_t d : base_dofs) {
    if (d >= 0 && static_cast<std::size_t>(d) < ndof) {
      collision_base_dofs_.push_back(static_cast<int>(d));
    }
  }

  // Phase 3 — predictive Cartesian scratch + params. dof_blocked_ marks
  // the base dofs so the arm Jacobian never realises an EE delta by "moving the
  // base" (which the collision FK zeroes anyway).
  q_predict_.assign(ndof, 0.0);
  dq_.assign(ndof, 0.0);
  dof_blocked_.assign(ndof, 0);
  for (const int d : collision_base_dofs_) {
    dof_blocked_[static_cast<std::size_t>(d)] = 1;
  }
  collision_ee_link_ = static_cast<int>(this->get_parameter("collision_ee_link_index").as_int());
  collision_predict_lambda_ = this->get_parameter("collision_predict_lambda").as_double();
  collision_predict_margin_growth_m_ =
      this->get_parameter("collision_predict_margin_growth_m").as_double();
  collision_predict_max_steps_ =
      static_cast<std::size_t>(this->get_parameter("collision_predict_max_steps").as_int());
  if (collision_ee_link_ >= 0 &&
      static_cast<std::size_t>(collision_ee_link_) >= collision_model_.n_links) {
    collision_ee_link_ = -1;  // out of range → disable predictive (fail-safe to reactive)
  }
  // The EE Jacobian walks root -> ee and gives a mimic link no column of its
  // own (`jacobian_dls_step`); a mimic ON that chain would move the EE without
  // a column, so the prediction would be wrong. No finger is on it (both jaws
  // are leaves); refuse rather than predict wrongly.
  for (int l = collision_ee_link_; l >= 0;
       l = collision_model_.parent[static_cast<std::size_t>(l)]) {
    if (is_mimic(collision_model_.joint_kind[static_cast<std::size_t>(l)])) {
      RCLCPP_ERROR(this->get_logger(),
                   "collision_ee_link_index %d has a mimic link on its chain; the EE Jacobian "
                   "cannot predict through a mimic joint",
                   collision_ee_link_);
      envelope_loaded_ = false;
      return CallbackReturn::FAILURE;
    }
  }

  joint_name_to_dof_.clear();
  for (std::size_t i = 0; i < collision_joint_names_.size() && i < ndof; ++i) {
    joint_name_to_dof_.emplace(collision_joint_names_[i], static_cast<int>(i));
  }
  collision_fk_dofs_.clear();
  for (const int d : collision_model_.dof_index) {
    // Once per column: a mimic link shares its leader's column.
    if (d >= 0 && static_cast<std::size_t>(d) < ndof &&
        std::find(collision_fk_dofs_.begin(), collision_fk_dofs_.end(), d) ==
            collision_fk_dofs_.end()) {
      collision_fk_dofs_.push_back(d);
    }
  }
  const bool seed_ready = !collision_joint_names_.empty() && !collision_fk_dofs_.empty();
  if (self_collision_enabled_ || world_voxel_enabled_) {
    RCLCPP_INFO(this->get_logger(),
                "velocity+cartesian collision: %s (joint_names=%zu, fk_dofs=%zu, dt=%gs, "
                "state_deadline=%gs)",
                seed_ready ? "armed"
                           : "INACTIVE (no collision_joint_names — velocity/cartesian "
                             "chunks will be passed by geometry; plumb the param to cover)",
                collision_joint_names_.size(), collision_fk_dofs_.size(), collision_seed_dt_s_,
                collision_state_deadline_s_);
  }

  // ADR-0121: last, once every open surface (collision model, grasp allowlist, #349, K2) is
  // configured. Never fails configure.
  load_exemption_provider();
  workspace_ingest_enabled_ = workspace_contact_enabled_ || provider_ != nullptr;

  safe_pub_ =
      this->create_publisher<openral_msgs::msg::ActionChunk>("/openral/safe_action", chunk_qos());
  estop_pub_ = this->create_publisher<std_msgs::msg::Empty>("/openral/estop", estop_qos());
  failure_pub_ = this->create_publisher<openral_msgs::msg::FailureTrigger>(
      "/openral/failure/safety", failure_qos());
  // ADR-0096 — additive observability publisher. It gates nothing: every
  // enforcement decision below is taken exactly as it was before, and this
  // only reports the decision that was already made.
  status_pub_ = this->create_publisher<openral_msgs::msg::SafetyStatus>("/openral/safety_status",
                                                                        safety_status_qos());
  diagnostics_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", rclcpp::QoS(rclcpp::KeepLast(1)));

  candidate_sub_ = this->create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos(),
      std::bind(&SafetyKernelLifecycleNode::on_candidate_action, this, std::placeholders::_1));
  estop_sub_ = this->create_subscription<std_msgs::msg::Empty>(
      "/openral/estop", estop_qos(),
      std::bind(&SafetyKernelLifecycleNode::on_external_estop, this, std::placeholders::_1));

  // Subscribe /joint_states only when a geometric check is enabled
  // and the joint-name map is plumbed (otherwise there is nothing to seed).
  if ((self_collision_enabled_ || world_voxel_enabled_ || attached_collision_enabled_ ||
       force_observer_enabled_) &&
      !collision_joint_names_.empty()) {
    rclcpp::QoS js_qos(rclcpp::KeepLast(1));
    js_qos.best_effort();
    js_qos.durability_volatile();
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", js_qos,
        std::bind(&SafetyKernelLifecycleNode::on_joint_state, this, std::placeholders::_1));
  }

  if (world_voxel_enabled_) {
    rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
    voxel_qos.reliable();
    voxel_qos.durability_volatile();
    voxel_sub_ = this->create_subscription<openral_msgs::msg::OccupancyVoxels>(
        "/openral/world_voxels", voxel_qos,
        std::bind(&SafetyKernelLifecycleNode::on_world_voxels, this, std::placeholders::_1));
  }
  if (attached_collision_enabled_ || grasp_allowance_enabled_) {
    // WorldStateStamped is published RELIABLE+VOLATILE+KL=1 at 30 Hz; the kernel
    // consumes the bounded attached_objects array (grasped payloads) and the
    // declarations riding beside it. The grasp exemption needs it even with
    // attached checking off: its region and handover payload live there.
    rclcpp::QoS world_state_qos(rclcpp::KeepLast(1));
    world_state_qos.reliable();
    world_state_qos.durability_volatile();
    world_state_sub_ = this->create_subscription<openral_msgs::msg::WorldStateStamped>(
        "/openral/world_state_fast", world_state_qos,
        std::bind(&SafetyKernelLifecycleNode::on_world_state, this, std::placeholders::_1));
  }
  if (attached_collision_enabled_) {
    rclcpp::QoS attachment_ack_qos(rclcpp::KeepLast(1));
    attachment_ack_qos.reliable();
    attachment_ack_qos.transient_local();
    attachment_applied_pub_ = this->create_publisher<std_msgs::msg::UInt64>(
        "/openral/attachment_state_applied", attachment_ack_qos);
  }

  if (force_observer_enabled_) {
    // Disclosure, not a gate: the estimate the collision tier judges, for
    // calibration (#373 load-cell go/no-go) and incident review. Sensor QoS.
    rclcpp::QoS ext_qos(rclcpp::KeepLast(5));
    ext_qos.best_effort();
    ext_qos.durability_volatile();
    external_torque_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "/openral/safety/external_torque", ext_qos);
  }

  estop_reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/openral/estop_reset", std::bind(&SafetyKernelLifecycleNode::on_estop_reset, this,
                                        std::placeholders::_1, std::placeholders::_2));

  diagnostics_timer_ = this->create_wall_timer(
      std::chrono::seconds(1), std::bind(&SafetyKernelLifecycleNode::publish_diagnostics, this));

  return CallbackReturn::SUCCESS;
}

SafetyKernelLifecycleNode::CallbackReturn
SafetyKernelLifecycleNode::on_activate(const rclcpp_lifecycle::State& state) {
  // rclcpp_lifecycle::LifecycleNode::on_activate() default activates all
  // registered managed publishers; we still call it via the base.
  clear_workspace_commands(ResetReason::kActivate);
  safe_pub_->on_activate();
  estop_pub_->on_activate();
  failure_pub_->on_activate();
  diagnostics_pub_->on_activate();
  status_pub_->on_activate();
  // Hazard-log HZ-0096-1 mitigation 1 — publish a fresh SafetyStatus on
  // EVERY activation, not only on the next fault. TRANSIENT_LOCAL means a
  // consumer that was already connected before this process restarted keeps
  // trusting the value the *previous* publisher left behind; an
  // activation-time publish overwrites that stale sample within one
  // activation cycle instead of waiting for the next real event. A latch
  // that survived a deactivate→activate cycle is reported as it stands —
  // recovery is never implied by activation.
  status_msg_.latched = fault_latch_;
  if (!fault_latch_) {
    status_msg_.drop_reason = openral_msgs::msg::SafetyStatus::DROP_NONE;
    status_msg_.detail = "kernel activated";
    status_msg_.rskill_id.clear();
    status_msg_.trace_id.clear();
  }
  publish_safety_status_now();
  return rclcpp_lifecycle::LifecycleNode::on_activate(state);
}

SafetyKernelLifecycleNode::CallbackReturn
SafetyKernelLifecycleNode::on_deactivate(const rclcpp_lifecycle::State& state) {
  safe_pub_->on_deactivate();
  estop_pub_->on_deactivate();
  failure_pub_->on_deactivate();
  diagnostics_pub_->on_deactivate();
  status_pub_->on_deactivate();
  return rclcpp_lifecycle::LifecycleNode::on_deactivate(state);
}

SafetyKernelLifecycleNode::CallbackReturn
SafetyKernelLifecycleNode::on_cleanup(const rclcpp_lifecycle::State& /*state*/) {
  diagnostics_timer_.reset();
  voxel_grid_.provider_gate = ProviderGate{};
  provider_.reset();
  provider_links_.clear();
  provider_status_ = "none";
  estop_reset_srv_.reset();
  candidate_sub_.reset();
  estop_sub_.reset();
  voxel_sub_.reset();
  world_state_sub_.reset();
  joint_state_sub_.reset();
  external_torque_pub_.reset();
  q_meas_received_ = false;
  safe_pub_.reset();
  estop_pub_.reset();
  failure_pub_.reset();
  status_pub_.reset();
  attachment_applied_pub_.reset();
  diagnostics_pub_.reset();
  envelope_loaded_ = false;
  fault_latch_ = false;
  chunks_passed_ = 0;
  chunks_dropped_ = 0;
  // Must reset with the other two: `scaled` is documented as a subset of
  // `passed`, and carrying it across a cleanup while they restart at 0 makes
  // `scaled > passed` reachable on /diagnostics.
  chunks_scaled_ = 0;
  last_logged_scale_ = 1.0;
  last_drop_reason_.clear();
  grasp_allowance_enabled_ = false;
  grasp_allowlist_.reset();
  grasp_region_ = GraspTargetRegion{};
  voxel_grid_.grasp_region = GraspTargetRegion{};
  grasp_declaration_stamp_ns_ = 0;
  grasp_declaration_timeout_s_ = 0.0;
  grasp_declaration_target_.clear();
  grasp_retired_.clear();
  grasp_retired_overflow_logged_ = false;
  tracked_retired_.clear();
  tracked_n_ = 0;
  cover_pass_n_ = 0;
  tracked_have_revision_ = false;
  tracked_revision_ = 0;
  tracked_refusal_reason_.clear();
  tracked_set_frame_.clear();
  tracked_grasp_enabled_ = false;
  tracked_grasp_ = TrackedGraspTable{};
  voxel_grid_.tracked_grasp = nullptr;
  tracked_cover_dirty_ = true;
  tracked_decl_active_ = false;
  tracked_decl_key_.clear();
  tracked_grasp_logged_n_ = static_cast<std::size_t>(-1);
  workspace_explained_enabled_ = false;
  workspace_explained_ = WorkspaceExplanation{};
  support_cells_marked_ = 0;
  support_cells_dirty_ = true;
  voxel_grid_.workspace_explained = nullptr;
  static_cells_marked_ = 0;
  static_cells_for_ = VoxelGrid{};
  static_cells_for_valid_ = false;
  static_cells_lattice_ok_ = false;
  workspace_explained_logged_n_ = static_cast<std::size_t>(-1);
  bound_.fill(BoundPayload{});
  grasp_latched_tracked_ = false;
  grasp_latched_epoch_ = 0;
  grasp_wrong_claim_ = false;
  grasp_region_refusal_reason_.clear();
  grasp_region_refusal_target_.clear();
  grasp_latched_region_ = GraspTargetRegion{};
  grasp_latched_target_.clear();
  grasp_latched_stamp_ns_ = 0;
  grasp_latched_ = false;
  grasp_latched_moved_warned_ = false;
  workspace_contact_enabled_ = false;
  workspace_ingest_enabled_ = false;
  workspace_links_.reset();
  workspace_region_ = WorkspaceContactRegion{};
  voxel_grid_.workspace_region = WorkspaceContactRegion{};
  workspace_target_.clear();
  workspace_refusal_reason_.clear();
  workspace_pairs_total_ = 0;
  clear_workspace_commands(ResetReason::kCleanup);
  // Drop the remembered status so the next activation's publish is never
  // suppressed by the transition gate (HZ-0096-1 mitigation 1).
  status_msg_ = openral_msgs::msg::SafetyStatus{};
  // Drain the BatchSpanProcessor before we release the node — anything
  // emitted during the final tick must reach the collector or the
  // dashboard's Safety ledger will show stale state on the next launch.
  otel::shutdown_tracing();
  return CallbackReturn::SUCCESS;
}

SafetyKernelLifecycleNode::CallbackReturn
SafetyKernelLifecycleNode::on_shutdown(const rclcpp_lifecycle::State& state) {
  return on_cleanup(state);
}

void SafetyKernelLifecycleNode::on_candidate_action(
    const openral_msgs::msg::ActionChunk::SharedPtr msg) {
  if (msg == nullptr) {
    return;
  }

  // Resume the producer's trace if the chunk carries a W3C traceparent
  // in `trace_id` ("OTel context is the truth; ROS fields
  // are set from it"). Empty / malformed values give us a root span,
  // which still flows to the dashboard's Safety card.
  auto parent_ctx = otel::extract_parent_context(msg->trace_id);
  auto ctx_scope = opentelemetry::context::RuntimeContext::Attach(parent_ctx);

  auto span_tracer = otel::tracer();
  opentelemetry::trace::StartSpanOptions span_opts;
  span_opts.kind = opentelemetry::trace::SpanKind::kInternal;
  auto span =
      span_tracer->StartSpan(otel::kSafetyCheckSpanName,
                             {
                                 {"safety.check_name", "envelope"},
                                 {"safety.kernel", otel::kSafetyKernelValue},
                                 // msg lives for the whole callback; its rskill_id storage
                                 // outlives span->End() below so the c_str() pointer is valid.
                                 // Key is `rskill.id` (semconv.RSKILL_ID) — the dashboard's
                                 // Identity card + _IDENTITY_KEYS latch this short-prefix form;
                                 // the legacy `openral.skill.id` is not read anywhere.
                                 {"rskill.id", msg->rskill_id.c_str()},
                             },
                             span_opts);
  auto span_scope = opentelemetry::trace::Scope(span);

  // #349: a refused chunk (any drop path, latching or not) since the last
  // candidate ends the previous-command chain — the robot's command is no
  // longer known to be the last accepted row. Checked here, once, so no drop
  // path can forget to do it.
  if (chunks_dropped_ != workspace_cmd_drops_seen_) {
    workspace_cmd_drops_seen_ = chunks_dropped_;
    clear_workspace_commands(ResetReason::kRefusal);
  }

  if (fault_latch_) {
    ++chunks_dropped_;
    last_drop_reason_ = "estop_latched";
    span->SetAttribute("safety.severity", "warn");
    span->SetAttribute("safety.drop_reason", "estop_latched");
    span->End();
    return;
  }
  if (!envelope_loaded_) {
    // No envelope; every chunk is a failure but we treat it as a
    // configuration error rather than a runtime estop trigger — the
    // operator needs to know the kernel is not yet armed.
    ++chunks_dropped_;
    last_drop_reason_ = "envelope_unconfigured";
    // ADR-0096 — this path was silent end to end before: no
    // /openral/safe_action, no /openral/estop, no FailureTrigger, only a span
    // attribute and a /diagnostics key-value. The operator now sees WHY
    // nothing is moving.
    set_safety_status(false, openral_msgs::msg::SafetyStatus::DROP_ENVELOPE_UNCONFIGURED,
                      "no envelope loaded; the kernel is not armed", msg->rskill_id, msg->trace_id);
    span->SetAttribute("safety.severity", "warn");
    span->SetAttribute("safety.drop_reason", "envelope_unconfigured");
    span->End();
    return;
  }

  ChunkView view{};
  view.control_mode = msg->control_mode;
  view.horizon = msg->horizon;
  view.n_dof = msg->n_dof;
  view.flat_data = msg->flat.empty() ? nullptr : msg->flat.data();
  view.flat_size = msg->flat.size();
  view.cartesian_delta_scale =
      msg->cartesian_delta_scale.empty() ? nullptr : msg->cartesian_delta_scale.data();
  view.cartesian_delta_scale_size = msg->cartesian_delta_scale.size();
  view.ee_name = msg->ee_name.data();
  view.ee_name_size = msg->ee_name.size();

  const auto result = validate(view, envelope_);
  if (result) {
    // Geometric collision over the chunk horizon (self + world).
    // Runs only for absolute joint-position chunks (the rows are full joint
    // configs FK can place). Allocation-free: FK reuses the pre-sized scratch.
    const bool geom_enabled =
        self_collision_enabled_ || world_voxel_enabled_ || attached_collision_enabled_;
    const auto mode = static_cast<ControlMode>(view.control_mode);
    // Smallest clearance ABOVE its own gate margin anywhere in this chunk's
    // sweep (#188). Slack, not raw distance: every check carries a different
    // margin and predictive steps inflate theirs with look-ahead depth, so
    // slack is the only margin-agnostic comparison.
    //
    // Pairs BELOW their margin that didn't trip are excluded: not tripping
    // means the pair is exempted (attach-time contact baseline, ACM row,
    // live place allowance) — letting an exemption drive the scale would
    // make a robot resting a payload on a shelf crawl for as long as it
    // holds it.
    double collision_min_slack_m = std::numeric_limits<double>::infinity();
    const bool is_position = (mode == ControlMode::kJointPosition);
    // Non-position chunks carry velocities / EE deltas, not joint
    // configs FK can place; reconstruct from the latest measured joint state.
    // Active only once the joint-name map is plumbed (`collision_joint_names`),
    // otherwise we cannot order the measured seed.
    const bool have_seed_map = !joint_name_to_dof_.empty();
    // Phase 2 — joint-velocity: reactive (measured config) + predictive integration.
    const bool is_velocity = (mode == ControlMode::kJointVelocity) && have_seed_map;
    // Phase 3 — Cartesian/twist (the arm mode for LIBERO/SIMPLER/DROID + the
    // robocasa arm chunk): rows are EE deltas/twists, not joint configs, so they
    // are checked REACTIVELY against the measured configuration (catches an arm
    // already in/at an obstacle; conservative — predictive IK/Jacobian is a later
    // phase). GRIPPER_*/COMPOSITE_MODE carry no arm geometry (scalar) → not here,
    // they fall through to safe_action (the companion arm chunk is checked).
    const bool is_cartesian =
        (mode == ControlMode::kCartesianPose || mode == ControlMode::kCartesianDelta ||
         mode == ControlMode::kCartesianTwist || mode == ControlMode::kBodyTwist) &&
        have_seed_map;
    // Position + velocity need full-dof rows (FK-placeable / integrable);
    // Cartesian uses only the measured seed, so it does not require full rows.
    const bool rows_full_dof = view.n_dof >= collision_required_dof_ && view.flat_data != nullptr;
    if (geom_enabled &&
        ((is_position && rows_full_dof) || (is_velocity && rows_full_dof) || is_cartesian)) {
      // World/seed availability gate: a chunk we cannot verify against a fresh
      // world model — or, for a seed-requiring mode, a fresh+complete measured
      // state — is dropped (fail-closed) but NOT latched; motion resumes once a
      // fresh input lands.
      const auto unavailable = [&](const char* reason, std::uint8_t drop_code) {
        ++chunks_dropped_;
        last_drop_reason_ = reason;
        Violation v;
        v.kind = ViolationKind::kController;
        v.set_field(reason);
        publish_failure_trigger(*msg, v);
        // ADR-0096 — the FailureTrigger above is VOLATILE, so a subscriber
        // that connects after the fact misses it entirely and it carries no
        // notion of current state. The latched status carries both, and its
        // DROP_* code says "an upstream input is unavailable", not "the
        // kernel is fault-latched" (it is not: no latch is set here).
        set_safety_status(false, drop_code, reason, msg->rskill_id, msg->trace_id);
        RCLCPP_WARN(this->get_logger(), "safety.world_unavailable reason=%s rskill_id=%s", reason,
                    msg->rskill_id.c_str());
        span->SetAttribute("safety.severity", "warn");
        span->SetAttribute("safety.drop_reason", reason);
        span->End();
      };
      // Velocity/Cartesian reconstruction needs a fresh, complete
      // measured seed; fail-closed otherwise.
      if ((is_velocity || is_cartesian) && !measured_state_fresh()) {
        unavailable("state_unavailable", openral_msgs::msg::SafetyStatus::DROP_STATE_UNAVAILABLE);
        return;
      }
      // ADR-0102 slot row. A slot-dispatched JOINT_POSITION chunk (the real
      // OpenArm: left arm, right arm as separate chunks per tick) is zero-padded
      // to full dof at the joints it does not own, so FK-ing the row as-is
      // places the OTHER arm at a phantom all-zero pose: a target that hits the
      // other arm where it really is passes, and one clear of it can false-stop.
      // `joint_names` says which joints the row commands; every other FK dof
      // takes its measured value, which therefore must be fresh and complete.
      // A row naming a joint the kernel cannot place is refused outright.
      // Velocity slots need nothing here: their zero padding already means
      // "hold the measured pose" in the integration above.
      bool slot_row = false;
      bool tick_partner = false;
      if (is_position && !msg->joint_names.empty()) {
        std::fill(slot_commanded_.begin(), slot_commanded_.end(), std::uint8_t{0});
        for (const auto& name : msg->joint_names) {
          const auto it = joint_name_to_dof_.find(name);
          if (it == joint_name_to_dof_.end() ||
              static_cast<std::size_t>(it->second) >= slot_commanded_.size()) {
            unavailable("slot_joints_unmapped",
                        openral_msgs::msg::SafetyStatus::DROP_STATE_UNAVAILABLE);
            return;
          }
          slot_commanded_[static_cast<std::size_t>(it->second)] = 1;
        }
        for (const int d : collision_fk_dofs_) {
          if (slot_commanded_[static_cast<std::size_t>(d)] == 0) {
            slot_row = true;
            break;
          }
        }
        if (slot_row && !measured_state_fresh()) {
          unavailable("state_unavailable", openral_msgs::msg::SafetyStatus::DROP_STATE_UNAVAILABLE);
          return;
        }
        // The HAL commits a tick's slots as ONE configuration, so a row is also
        // checked against the targets the tick's earlier (accepted) slots set
        // for the joints it leaves alone — both arms' targets at once. Keyed on
        // (runner_session_id, tick_index); an unstamped chunk gets the measured
        // check only.
        if (slot_row && msg->tick_index != 0) {
          if (msg->runner_session_id != tick_target_session_ ||
              msg->tick_index != tick_target_tick_) {
            std::fill(tick_target_set_.begin(), tick_target_set_.end(), std::uint8_t{0});
            tick_target_session_ = msg->runner_session_id;
            tick_target_tick_ = msg->tick_index;
          }
          for (std::size_t d = 0; d < tick_target_set_.size(); ++d) {
            if (tick_target_set_[d] != 0 && slot_commanded_[d] == 0) {
              tick_partner = true;
              break;
            }
          }
        }
      }
      if (world_voxel_enabled_) {
        const bool fresh = voxel_received_ && !voxel_overflow_ &&
                           (this->now() - voxel_stamp_).seconds() <= world_voxel_deadline_s_;
        if (!fresh) {
          unavailable(voxel_overflow_ ? "voxel_overflow" : "voxel_unavailable",
                      voxel_overflow_ ? openral_msgs::msg::SafetyStatus::DROP_VOXEL_OVERFLOW
                                      : openral_msgs::msg::SafetyStatus::DROP_VOXEL_UNAVAILABLE);
          return;
        }
        // Data age: how old the WORLD in the grid is, not how long ago the grid
        // arrived. A grid received 10 ms ago can describe the cell as it was a
        // second ago; the receipt deadline above is blind to that.
        if (voxel_source_known_) {
          const double data_age_ms = (this->now() - voxel_source_stamp_).seconds() * 1e3;
          span->SetAttribute("safety.world_voxel_data_age_ms", data_age_ms);
          if (data_age_ms > world_voxel_data_age_budget_s_ * 1e3) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "safety.voxel_stale data_age_ms=%.0f budget_ms=%.0f", data_age_ms,
                                 world_voxel_data_age_budget_s_ * 1e3);
            unavailable("voxel_stale", openral_msgs::msg::SafetyStatus::DROP_VOXEL_UNAVAILABLE);
            return;
          }
        } else {
          unavailable("voxel_stale", openral_msgs::msg::SafetyStatus::DROP_VOXEL_UNAVAILABLE);
          return;
        }
      }
      if (attached_collision_enabled_) {
        // A grasped payload we cannot verify (never received, over-capacity /
        // unknown-link, or stale) is fail-closed: we do not know what the robot
        // is carrying, so we must not certify the motion. An empty-but-fresh
        // attachment set is valid (nothing carried).
        const double attached_age_s = (this->now() - attached_stamp_).seconds();
        const bool fresh = attached_received_ && !attached_overflow_ && attached_age_s >= 0.0 &&
                           attached_age_s <= attached_collision_deadline_s_;
        if (!fresh) {
          unavailable(attached_overflow_ ? "attached_overflow" : "attached_unavailable",
                      attached_overflow_
                          ? openral_msgs::msg::SafetyStatus::DROP_ATTACHED_OVERFLOW
                          : openral_msgs::msg::SafetyStatus::DROP_ATTACHED_UNAVAILABLE);
          return;
        }
        if ((attached_contact_snapshot_pending_ || attached_contact_active_) &&
            !measured_state_fresh()) {
          unavailable("state_unavailable", openral_msgs::msg::SafetyStatus::DROP_STATE_UNAVAILABLE);
          return;
        }
      }

      const auto link_name = [this](int idx) -> std::string {
        if (idx >= 0 && static_cast<std::size_t>(idx) < collision_link_names_.size()) {
          return collision_link_names_[static_cast<std::size_t>(idx)];
        }
        return std::string("link_") + std::to_string(idx);
      };
      const auto attached_label = [this](int idx) -> std::string {
        if (idx >= 0 && static_cast<std::size_t>(idx) < attached_labels_.size() &&
            !attached_labels_[static_cast<std::size_t>(idx)].empty()) {
          return std::string("attached:") + attached_labels_[static_cast<std::size_t>(idx)];
        }
        return std::string("attached_") + std::to_string(idx);
      };
      // A trip's named geometry as evidence text; only on a report or a log line.
      const auto name_str = [&](WsName n) -> std::string {
        switch (n.tag) {
        case 'a':
          return attached_label(n.idx);
        case 'v':
          return std::string("voxel_") + std::to_string(n.idx);
        case 'p':
          return "place:" + place_declaration_target_ + "#" + std::to_string(n.idx);
        case 'f':
          return "force:" + place_declaration_target_;
        default:
          return link_name(n.idx);
        }
      };
      const auto trip_str = [&](const WsTrip& t) -> std::string {
        return t.set ? std::string(t.kind) + " " + name_str(t.a) + " " + name_str(t.b)
                     : std::string();
      };
      // Report one collision hit. `hit` supplies BOTH identity and distance,
      // so E-stop evidence always describes a single geometry pair:
      // hit.min_distance is that pair's own surface distance, never the
      // sweep-wide minimum. The sweep minimum (may belong to an exempted
      // pair, e.g. an attached payload's attach-time contact baseline) is
      // logged separately under sweep_min_distance_m and its own span
      // attribute, and stays out of min_distance_m.
      //
      // Evidence also carries q_fk_, the configuration the hit was measured
      // at. `report` is reached only from check_config right after FK, so
      // this is the only geometry the verdict can be about. A predicted
      // step's configuration is reconstructible from no other artifact
      // (depends on the kernel's own DLS Jacobian, lambda, dt) — the
      // drawer-opening stop that opened this issue was adjudicated against
      // the *measured* joints, where the two named links sit 53 mm apart.
      const auto report = [&](const char* kind, WsName na, WsName nb, int step,
                              const CollisionHit& hit) {
        if (ws_dry_run_) {
          // #349 brake dry run: nothing is reported. The first trip of each pass is
          // recorded without allocating (#349 review F): the as-is pass's is the
          // verdict if the brake does not answer it, the relaxed pass's says why.
          WsTrip& t = ws_speed_override_ ? ws_refused_trip_ : ws_first_trip_;
          if (!t.set) {
            t = WsTrip{true,
                       kind,
                       na,
                       nb,
                       step,
                       hit,
                       voxel_grid_.workspace_region.hand_speed_mps,
                       voxel_grid_.workspace_region.hand_lead_m,
                       ws_speed_measured_row_};
            if (!ws_speed_override_) {
              std::copy(q_fk_.begin(), q_fk_.end(), ws_trip_q_.begin());
            }
          }
          return;
        }
        const std::string a = name_str(na);
        const std::string b = name_str(nb);
        ++chunks_dropped_;
        // Advisory band (#176): a declared payload's own receptacle contact is
        // refused, not latched — chunk dropped, skill may retry, instead of an
        // operator /openral/estop_reset over a contact measured in millimetres.
        //
        // Bounded three ways (any failing gives today's stop): hit.advisory
        // is set only for an attached payload inside its live declaration,
        // within one voxel of the approach allowance; refusals are capped;
        // every other check here reaches report with advisory == false.
        if (hit.advisory && advisory_refusals_ < place_advisory_max_consecutive_) {
          ++advisory_refusals_;
          last_drop_reason_ = "collision_advisory";
          publish_collision_failure(*msg, kind, a, b, step, hit, q_fk_);
          RCLCPP_WARN(this->get_logger(),
                      "safety.collision_advisory kind=%s a=%s b=%s step=%d min_distance_m=%g "
                      "sweep_min_distance_m=%g place_target=%s consecutive=%llu/%llu "
                      "(action refused, no latch)",
                      kind, a.c_str(), b.c_str(), step, hit.min_distance, hit.sweep_min_distance,
                      place_declaration_target_.c_str(),
                      static_cast<unsigned long long>(advisory_refusals_),
                      static_cast<unsigned long long>(place_advisory_max_consecutive_));
          span->SetAttribute("safety.severity", "advisory");
          span->SetAttribute("safety.drop_reason", "collision_advisory");
          span->SetAttribute("safety.violation_value", hit.min_distance);
          span->SetAttribute("safety.sweep_min_distance_m", hit.sweep_min_distance);
          span->SetAttribute("safety.place_allowance_active", hit.place_allowance_active);
          span->SetAttribute("safety.advisory_consecutive",
                             static_cast<int64_t>(advisory_refusals_));
          span->End();
          return;
        }
        last_drop_reason_ = "collision";
        fault_latch_ = true;
        last_estop_at_ = std::chrono::steady_clock::now();
        publish_collision_failure(*msg, kind, a, b, step, hit, q_fk_);
        set_safety_status(true, openral_msgs::msg::SafetyStatus::KIND_COLLISION, kind,
                          msg->rskill_id, msg->trace_id);
        std_msgs::msg::Empty estop_msg;
        estop_pub_->publish(estop_msg);
        // `place_allowance_active` is disclosure, never justification: the stop
        // happened, and the distance quoted is the pair's true one. It says the
        // margin that pair was gated against had been reduced by a live place
        // declaration, so an incident review can tell that case apart from an
        // ordinary stop without re-deriving it (CLAUDE.md §1.4). Under K2 the tracked table,
        // not the (cleared) legacy region, is the grasp exemption (as in the FailureTrigger).
        const bool grasp_active =
            tracked_grasp_enabled_ ? tracked_grasp_.n_objects > 0 : voxel_grid_.grasp_region.valid;
        RCLCPP_ERROR(this->get_logger(),
                     "safety.collision kind=%s a=%s b=%s step=%d min_distance_m=%g "
                     "sweep_min_distance_m=%g mode=%u rskill_id=%s place_allowance_active=%d "
                     "place_target=%s depth_is_box_bound=%d grasp_exemption_active=%d "
                     "grasp_target=%s workspace_contact_active=%d hand_speed_mps=%g "
                     "hand_lead_m=%g hand_speed_source=%s workspace_denied=%d",
                     kind, a.c_str(), b.c_str(), step, hit.min_distance, hit.sweep_min_distance,
                     static_cast<unsigned>(view.control_mode), msg->rskill_id.c_str(),
                     static_cast<int>(hit.place_allowance_active),
                     hit.place_allowance_active ? place_declaration_target_.c_str() : "",
                     static_cast<int>(hit.depth_is_box_bound), static_cast<int>(grasp_active),
                     grasp_active ? grasp_declaration_target_.c_str() : "",
                     static_cast<int>(voxel_grid_.workspace_region.valid),
                     voxel_grid_.workspace_region.hand_speed_mps,
                     voxel_grid_.workspace_region.hand_lead_m,
                     ws_speed_measured_row_ ? "measured" : "commanded", hit.workspace_denied);
        if (voxel_grid_.workspace_region.valid) {
          span->SetAttribute("safety.workspace_hand_speed_mps",
                             voxel_grid_.workspace_region.hand_speed_mps);
          span->SetAttribute("safety.workspace_hand_speed_source",
                             ws_speed_measured_row_ ? "measured" : "commanded");
          span->SetAttribute("safety.workspace_hand_lead_m",
                             voxel_grid_.workspace_region.hand_lead_m);
        }
        span->SetAttribute("safety.severity", "violation");
        span->SetAttribute("safety.drop_reason", "collision");
        span->SetAttribute("safety.collision_mode", static_cast<int64_t>(view.control_mode));
        span->SetAttribute("safety.violation_value", hit.min_distance);
        span->SetAttribute("safety.sweep_min_distance_m", hit.sweep_min_distance);
        span->SetAttribute("safety.place_allowance_active", hit.place_allowance_active);
        // Disclosure, never a gate: the reported depth is the OBB's bound
        // rather than a measurement of the hulls the evidence names. An
        // adjudicator that subtracts this number from a mesh-level probe must
        // not read it as the hull depth (#191/#202, 2026-09-04 battery).
        span->SetAttribute("safety.depth_is_box_bound", hit.depth_is_box_bound);
        span->AddEvent(otel::kSafetyViolationEventName, {{"safety.kind", kind}});
        span->End();
      };

      // FK one configuration `q` (length n_dof) and run the enabled geometric
      // checks; report + estop and return true on the first hit. Shared by the
      // position and velocity paths so they can never diverge.
      const std::size_t robot_ndof = q_fk_.size();  // FK dof span (envelope n_dof)
      // FK a full robot-dof configuration `q` in the base_link frame: copy + zero
      // the mobile-base dofs so the arm capsules land in the same frame as the
      // base-relative world/voxel grid (otherwise the base's world pose pushes
      // the arm metres outside the local map). No-op for fixed-base arms. Leaves
      // the result in collision_scratch_ (reused by the predictive Jacobian).
      const auto fk_config = [&](const double* q) {
        std::copy(q, q + robot_ndof, q_fk_.begin());
        for (const int d : collision_base_dofs_) {
          q_fk_[static_cast<std::size_t>(d)] = 0.0;
        }
        forward_kinematics(collision_model_, q_fk_.data(), robot_ndof, collision_scratch_);
      };
      if (attached_contact_snapshot_pending_ || attached_contact_active_ ||
          support_witness_live_ != 0) {
        fk_config(q_meas_.data());
        if (attached_contact_snapshot_pending_ || attached_contact_active_) {
          attached_contact_active_ = update_attached_voxel_contacts(
              attached_model_, collision_scratch_, voxel_grid_, attached_contact_mask_.data(),
              attached_contact_distance_.data(), attached_contact_mask_.size(),
              attached_contact_distance_.size(), attached_contact_snapshot_pending_);
          attached_contact_snapshot_pending_ = false;
        }
        // Refresh the witness latch against the MEASURED configuration only —
        // a predicted pose must never keep an exemption alive. Separation kills
        // it here, permanently, until World State attests a new contact.
        const std::uint8_t before = support_witness_live_;
        support_witness_live_ =
            update_support_contact_witnesses(attached_model_, collision_scratch_, voxel_grid_,
                                             support_witness_live_, attached_collision_margin_m_);
        if (support_witness_live_ != before) {
          RCLCPP_INFO(this->get_logger(), "safety.support_witness_separated live=0x%x was=0x%x",
                      static_cast<unsigned>(support_witness_live_), static_cast<unsigned>(before));
        }
      }
      voxel_grid_.support_witness_live = support_witness_live_;
      // Re-evaluated per candidate action, not only when a world state lands:
      // the declaration's own backstop is what stops an allowance outliving the
      // goal that justified it if the producer stalls (HZ-0097-3/4).
      voxel_grid_.place_region = place_declaration_live() ? place_region_ : PlaceApproachRegion{};
      // ADR-0115 draft: the grasp exemption, re-evaluated per candidate for the
      // same reason. During the handover (the declared object is attached) it
      // lives only while that payload's origin — FK of the MEASURED
      // configuration, never a predicted one — is inside the region; once it
      // leaves, the declaration retires for good. No fresh measured state means
      // no exemption for this candidate (not a retirement).
      bool grasp_live = grasp_allowance_enabled_ && grasp_declaration_live();
      if (grasp_live && grasp_region_.handover) {
        if (!measured_state_fresh() || grasp_region_.object_index >= attached_model_.n_objects ||
            attached_model_.objects[grasp_region_.object_index].attach_link < 0) {
          grasp_live = false;
        } else {
          fk_config(q_meas_.data());
          const AttachedObject& obj = attached_model_.objects[grasp_region_.object_index];
          const Transform& link =
              collision_scratch_.link_world[static_cast<std::size_t>(obj.attach_link)];
          const Vec3& p = obj.pose_in_link.t;
          const Vec3 origin{link.t.x + link.r[0] * p.x + link.r[1] * p.y + link.r[2] * p.z,
                            link.t.y + link.r[3] * p.x + link.r[4] * p.y + link.r[5] * p.z,
                            link.t.z + link.r[6] * p.x + link.r[7] * p.y + link.r[8] * p.z};
          if (!grasp_region_contains(grasp_region_, origin)) {
            retire_grasp_declaration("handover_exit");
            grasp_live = false;
          }
        }
      }
      voxel_grid_.grasp_region = grasp_live ? grasp_region_ : GraspTargetRegion{};
      const bool region_live = grasp_live;  // ADR-0115's verdict, before K2 re-sources it
      if (tracked_grasp_enabled_) {
        // #418 K2: the tracked table REPLACES the region as the exemption source
        // (the check reads only the table while it is valid, as it always is here).
        voxel_grid_.grasp_region = GraspTargetRegion{};
        voxel_grid_.tracked_grasp = &tracked_grasp_;
        grasp_live = fill_tracked_grasp() > 0;
        span->SetAttribute("safety.tracked_grasp_eligible",
                           static_cast<int64_t>(tracked_grasp_.n_objects));
      }
      span->SetAttribute("safety.grasp_exemption_active", grasp_live);
      // #349: the manipulation workspace, re-evaluated per candidate like the
      // grasp region. Its hand speed and lead are set per checked row below;
      // they are known only for a JOINT_POSITION chunk with a fresh measured
      // state (the lead's reference, and row 0's fallback). Every other mode
      // keeps them +inf, i.e. no exemption.
      const bool workspace_on = workspace_contact_enabled_ && workspace_live();
      voxel_grid_.workspace_region = workspace_on ? workspace_region_ : WorkspaceContactRegion{};
      if (workspace_explained_enabled_) {
        // #418 K7: the workspace is narrowed to explained cells, every candidate.
        voxel_grid_.workspace_explained = &workspace_explained_;
        const std::size_t explaining = fill_workspace_explained();
        span->SetAttribute("safety.workspace_explained_objects", static_cast<int64_t>(explaining));
      }
      if (payload_own_enabled_) {
        // #418 OT-10 amendment: a bound payload's measured cells cannot stop that payload.
        voxel_grid_.payload_own = &payload_own_;
        span->SetAttribute("safety.payload_own_armed", static_cast<int64_t>(fill_payload_own()));
      }
      // ADR-0121: hand the provider this candidate's read-only view. Every pointer is kernel-owned
      // and preallocated; the provider is asked about pairs only through the voxel check below.
      voxel_grid_.provider_gate.provider = provider_.get();
      voxel_grid_.provider_gate.links = provider_links_.empty() ? nullptr : provider_links_.data();
      voxel_grid_.provider_gate.links_n = provider_links_.size();
      voxel_grid_.provider_gate.row = 0;
      const ProviderGate* const provider_gate =
          provider_ != nullptr ? &voxel_grid_.provider_gate : nullptr;
      if (provider_ != nullptr) {
        std::size_t n_view = 0;
        for (std::size_t i = 0; i < tracked_n_ && n_view < provider_tracked_.size(); ++i) {
          const TrackedEntry& e = tracked_[i];
          if (tracked_retired_.contains(e.id_hash, e.epoch)) {
            continue;
          }
          TrackedObjectView& v = provider_tracked_[n_view++];
          v.id_hash = e.id_hash;
          v.epoch = e.epoch;
          v.live = e.live;
          v.released = e.released;
          v.target = e.target;
          v.contact = e.contact;
          v.exemptible = e.exemptible;
          v.provisional = e.provisional;
          v.cover_ok = e.cover_ok;
          v.visible_cells = e.visible_cells;
          v.confirmed_ns = e.confirmed_ns;
          v.support_z = e.support_z;
          v.floor_z = e.floor_z;
          v.declaration_key_hash = std::hash<std::string>{}(e.declaration_key);
          v.box = e.box;
        }
        const bool fresh = measured_state_fresh();
        // The kernel's own FK at the measured state, base dofs zeroed as its checks do.
        const bool fk_meas = fresh && q_meas_.size() == provider_q_fk_.size() &&
                             !provider_meas_scratch_.link_world.empty();
        if (fk_meas) {
          std::copy(q_meas_.begin(), q_meas_.end(), provider_q_fk_.begin());
          for (const int d : collision_base_dofs_) {
            provider_q_fk_[static_cast<std::size_t>(d)] = 0.0;
          }
          forward_kinematics(collision_model_, provider_q_fk_.data(), provider_q_fk_.size(),
                             provider_meas_scratch_);
        }
        CandidateView pview;
        pview.model = &collision_model_;
        pview.grid = world_voxel_enabled_ ? &voxel_grid_ : nullptr;
        pview.q_measured = fresh && !q_meas_.empty() ? q_meas_.data() : nullptr;
        pview.dq_measured = fresh && dq_meas_.size() == q_meas_.size() && !dq_meas_.empty()
                                ? dq_meas_.data()
                                : nullptr;
        pview.n_dof = q_meas_.size();
        pview.tracked = provider_tracked_.data();
        pview.n_tracked = n_view;
        pview.relaxable_links = voxel_grid_.provider_gate.links;
        pview.now_ns = this->now().nanoseconds();
        pview.position_mode = is_position;
        pview.control_period_s = collision_seed_dt_s_;
        pview.measured = fk_meas ? &provider_meas_scratch_ : nullptr;
        pview.region_max_age_ns = static_cast<std::int64_t>(grasp_region_max_age_s_ * 1e9);
        pview.tracked_frame_ok = tracked_n_ > 0 && refresh_tracked_cover();
        DeclarationView& dv = pview.declaration;
        dv.live = region_live;
        dv.refused = grasp_declaration_refused_;
        dv.wrong_claim = grasp_wrong_claim_;
        dv.tracked_active = tracked_decl_active_;
        dv.key_hash = tracked_decl_active_ ? tracked_decl_key_hash_ : 0;
        dv.stamp_ns = grasp_declaration_stamp_ns_;
        dv.timeout_s = grasp_declaration_timeout_s_;
        dv.grasp_region = region_live && grasp_region_.valid ? &grasp_region_ : nullptr;
        dv.workspace = workspace_live() ? &workspace_region_ : nullptr;
        dv.hands = provider_hands_valid_ ? &provider_hands_ : nullptr;
        provider_->begin_candidate(pview);
      }
      const bool workspace_speed_known = workspace_on && is_position && measured_state_fresh();
      workspace_speed_measured_ = false;  // set below for a position chunk with dq_meas_
      ws_hand_contact_ = false;
      ws_speed_measured_row_ = false;
      span->SetAttribute("safety.workspace_contact_active", workspace_on);
      std::uint64_t workspace_chunk_pairs = 0;
      std::int64_t provider_chunk_pairs = 0;  // ADR-0121, this candidate's tolerated pairs
      double workspace_chunk_min = std::numeric_limits<double>::infinity();
      // FK `q` then run the enabled checks at `margin + extra_margin` (extra>0 for
      // predictive steps, inflating with look-ahead depth). report + return true
      // on the first hit. `q` is a position row, the measured seed, or a
      // predicted Cartesian config.
      // Fold one check's sweep-wide minimum into the chunk's smallest slack
      // (#188). `m` is the margin THAT check was gated against, including the
      // predictive look-ahead inflation, so the comparison is like-for-like.
      //
      // A check that did not trip can still read a negative sweep minimum: an
      // EXEMPTED pair (live support witness, embedded attach-time residue, the
      // grasp target) sits inside the margin by construction and reaches the
      // sweep minimum at its true depth. The sweep keeps ONE minimum, so
      // discarding that negative would discard every non-exempt pair's graded
      // slack with it and send the chunk out at full rate beside a cell it must
      // not touch. Clamped to 0, an exempt contact reads as the band's slowest
      // rate: the most conservative reading short of a stop. A check that DID
      // trip reports and returns, so its clamped slack never scales anything.
      const auto note_slack = [&](const CollisionHit& h, double m) {
        const double slack = std::max(h.sweep_min_distance - m, 0.0);
        if (slack < collision_min_slack_m) {
          collision_min_slack_m = slack;
        }
      };
      // #349 disclosure of tolerated workspace pairs, never silent: count, tally,
      // WARN <= 1 Hz per link. `who` prefixes the link: "" for a hand link,
      // "payload@" for a payload carried by it.
      const auto disclose_workspace = [&](const CollisionHit& hit, int step, const char* who) {
        if (hit.workspace_pairs <= 0 || ws_quiet_) {
          return;
        }
        const double speed = voxel_grid_.workspace_region.hand_speed_mps;
        const double lead = voxel_grid_.workspace_region.hand_lead_m;
        workspace_chunk_pairs += static_cast<std::uint64_t>(hit.workspace_pairs);
        workspace_chunk_min = std::min(workspace_chunk_min, hit.workspace_min_distance);
        workspace_pairs_total_ += static_cast<std::uint64_t>(hit.workspace_pairs);
        workspace_min_distance_total_ =
            std::min(workspace_min_distance_total_, hit.workspace_min_distance);
        if (speed > workspace_speed_max_seen_) {
          workspace_speed_max_seen_ = speed;
          workspace_speed_max_measured_ = ws_speed_measured_row_;
        }
        workspace_lead_max_seen_ = std::max(workspace_lead_max_seen_, lead);
        const auto l = static_cast<std::size_t>(hit.workspace_link);
        const auto wall = std::chrono::steady_clock::now();
        if (l < workspace_last_warn_.size() &&
            wall - workspace_last_warn_[l] >= std::chrono::seconds(1)) {
          workspace_last_warn_[l] = wall;
          RCLCPP_WARN(this->get_logger(),
                      "safety.workspace_contact link=%s%s cell=%d depth_mm=%.1f "
                      "speed_mps=%.3f speed_source=%s max_mps=%g lead_m=%.4f "
                      "lead_allow_m=%.4f max_lead_m=%g pairs=%d step=%d target=%s",
                      who, link_name(hit.workspace_link).c_str(), hit.workspace_cell,
                      std::max(0.0, -hit.workspace_min_distance) * 1e3, speed,
                      ws_speed_measured_row_ ? "measured" : "commanded", workspace_max_speed_mps_,
                      lead, ws_lead_allow_m_, workspace_max_lead_m_, hit.workspace_pairs, step,
                      workspace_target_.c_str());
        }
        span->SetAttribute("safety.workspace_contact_pairs",
                           static_cast<int64_t>(workspace_chunk_pairs));
        span->SetAttribute("safety.workspace_contact_max_depth_m",
                           std::max(0.0, -workspace_chunk_min));
        span->SetAttribute("safety.workspace_hand_speed_mps", speed);
        span->SetAttribute("safety.workspace_hand_speed_source",
                           ws_speed_measured_row_ ? "measured" : "commanded");
        span->SetAttribute("safety.workspace_hand_lead_m", lead);
      };
      // ADR-0121: never silent. Every pair the provider tolerated is counted, the chunk's count is
      // a span attribute, and one WARN a second names the deepest.
      const auto disclose_provider = [&](const CollisionHit& hit, int step, const char* who) {
        if (hit.provider_pairs <= 0 || provider_ == nullptr) {
          return;
        }
        provider_pairs_total_ += static_cast<std::uint64_t>(hit.provider_pairs);
        provider_chunk_pairs += static_cast<std::int64_t>(hit.provider_pairs);
        span->SetAttribute("safety.exemption_provider_pairs", provider_chunk_pairs);
        const auto wall = std::chrono::steady_clock::now();
        if (wall - provider_last_warn_ >= std::chrono::seconds(1)) {
          provider_last_warn_ = wall;
          RCLCPP_WARN(this->get_logger(),
                      "safety.exemption_provider_contact provider=%s link=%s%s other=%d "
                      "depth_mm=%.1f pairs=%d step=%d",
                      provider_->name(), who, link_name(hit.provider_link).c_str(),
                      hit.provider_cell, -hit.provider_min_distance * 1e3, hit.provider_pairs,
                      step);
        }
      };
      // `workspace_prev`: the hand links' poses at the previous row of the same
      // construction (for row 0, the previously accepted command where fresh,
      // else the measured state); nullptr = speed and lead unknown.
      const auto check_config = [&](const double* q, int step, double extra_margin = 0.0,
                                    std::vector<Transform>* workspace_prev = nullptr) -> bool {
        if (voxel_grid_.workspace_region.valid) {
          const bool known = workspace_prev != nullptr;
          // The lead at `q` with the jaws held at measured: the arm's push only.
          if (known) {
            std::copy(q, q + q_lead_.size(), q_lead_.begin());
            for (const int d : workspace_jaw_cols_) {
              q_lead_[static_cast<std::size_t>(d)] = q_meas_[static_cast<std::size_t>(d)];
            }
            fk_config(q_lead_.data());
          }
          // Row `step` is executed `step` periods from now: by then a free hand may
          // have travelled `step` periods at the speed bound, so only the lead past
          // that allowance is a push (#349 r31: a chunk's last row 1.7 s out led
          // 0.48 m on a plain carry and braked every chunk). Row 0 (and the measured
          // state, step -1) gets no allowance: a blocked hand is re-judged against
          // the fresh measured pose at every chunk, so the executed push never
          // exceeds the lead bound plus one period. A hand already touching
          // (`ws_hand_contact_`) gets no allowance on any row (#349 review A: a blocked
          // hand does not travel with the command, and one open-loop chunk could push
          // it max_lead + s*v*T, ~0.18 m). A hand free now that meets something
          // mid-chunk keeps the allowance until the next chunk's re-check, which sees
          // it in contact: that push is bounded by one chunk's allowance.
          ws_lead_allow_m_ = step > 0 && !ws_hand_contact_
                                 ? workspace_max_speed_mps_ * workspace_control_period_s_ *
                                       static_cast<double>(step)
                                 : 0.0;
          voxel_grid_.workspace_region.hand_lead_m =
              !known               ? std::numeric_limits<double>::infinity()
              : ws_speed_override_ ? 0.0
                                   : std::max(0.0, workspace_hand_lead() - ws_lead_allow_m_);
          fk_config(q);
          // Row 0 with a fresh measured velocity: the measured speed only. Its
          // commanded step (from the previous command or the measured state) is the
          // chunk-boundary jump the arm smooths, already bounded by the lead (#349 r41:
          // reading it as speed held the arm still every chunk, 2013 brakes a run).
          // Rows s >= 1: the larger of the measured and the commanded row-to-row speed
          // (#349 review B: a fast command within a chunk must not read slow). Without a
          // fresh measured velocity the commanded speed governs every row.
          double speed =
              known && ws_speed_override_ ? 0.0 : std::numeric_limits<double>::infinity();
          ws_speed_measured_row_ = false;
          if (known && !ws_speed_override_) {
            const double commanded = workspace_hand_speed(*workspace_prev);  // advances prev
            speed = commanded;
            if (workspace_speed_measured_ &&
                (step <= 0 || workspace_measured_speed_mps_ >= commanded)) {
              speed = workspace_measured_speed_mps_;
              ws_speed_measured_row_ = true;
            }
          }
          voxel_grid_.workspace_region.hand_speed_mps = speed;
        } else {
          fk_config(q);
        }
        if (self_collision_enabled_) {
          // #349: two slow hands meeting inside the live workspace are tolerated
          // like a hand meeting an object there (WARN `link=self@<a> cell=<b index>`).
          voxel_grid_.provider_gate.row = step;
          const auto hit = check_self_collision(
              collision_model_, collision_scratch_, self_collision_margin_m_ + extra_margin,
              workspace_contact_enabled_ ? &voxel_grid_.workspace_region : nullptr, provider_gate);
          disclose_workspace(hit, step, "self@");
          disclose_provider(hit, step, "self@");
          // NOT folded into the graded band (#188), deliberately, since #191.
          // A robot's tightest self-pair is a property of how it is BUILT, not
          // of where it is going: on `panda_mobile`, `panda_link5` <->
          // `panda_link7` is never further apart than 22.72 mm at ANY of the
          // 14641 poses of the subspace that moves it — 0 of them clear a 50 mm
          // band, let alone the 100 mm one. Folding that in pins the scale at a
          // constant (0.21 at 100 mm / k=20, measured) and the world term, which
          // the chunk CAN act on, then only matters below 22 mm. That is not a
          // slowdown, it is a permanent speed limit wearing one's name. The
          // self-collision LATCH is untouched; only its contribution to the
          // velocity band is dropped, which restores the pre-#188 rate on the
          // self path and leaves the world path doing what #188 built it for.
          if (hit.hit) {
            report("self", WsName{'l', hit.link_a}, WsName{'l', hit.link_b}, step, hit);
            return true;
          }
        }
        if (world_voxel_enabled_) {
          // The band is handed in as extra SCAN width, not extra margin: a
          // cell still trips at the margin, but a cell that is merely inside
          // the band has to be visited for `sweep_min_distance` to see it at
          // all. Without this the window is sized by the margin alone and the
          // minimum jumps from "nothing in range" straight to a tripping cell —
          // which made this whole mechanism dead code until a real scene run
          // showed the scale going 1.0 → E-stop with nothing in between.
          voxel_grid_.provider_gate.row = step;
          const auto hit = check_voxel_collision(collision_model_, collision_scratch_, voxel_grid_,
                                                 world_voxel_margin_m_ + extra_margin,
                                                 collision_scale_proximity_m_);
          note_slack(hit, world_voxel_margin_m_ + extra_margin);
          disclose_workspace(hit, step, "");
          disclose_provider(hit, step, "");
          if (hit.hit) {
            report("world", WsName{'l', hit.link_a}, WsName{'v', hit.link_b}, step, hit);
            return true;
          }
        }
        if (attached_collision_enabled_ && attached_model_.n_objects > 0) {
          // Grasped payloads (ADR-0092): check the attach-link-composed payload
          // geometry against occupancy voxels and the robot's own links
          // (except the attach link + explicit touch links). The
          // collision_kind stays "self"/"world" to satisfy CollisionEvidence;
          // the attach:<object_id> label marks it as a payload hit.
          const double amargin = attached_collision_margin_m_ + extra_margin;
          // Legacy contact phase: without an attestation the kernel still can
          // not tell a legitimate support contact from a real one, so it keeps
          // skipping the predicted Cartesian steps. With a live witness it does
          // not need to — the exemption is pose-dependent and bounded — so the
          // predicted steps are checked too. That is strictly more checking.
          const bool contact_constrained_prediction =
              is_cartesian && attached_contact_active_ && step >= 0 && support_witness_live_ == 0;
          if (world_voxel_enabled_ && !contact_constrained_prediction) {
            const auto hit = check_attached_voxel_collision(collision_model_, attached_model_,
                                                            collision_scratch_, voxel_grid_,
                                                            amargin, collision_scale_proximity_m_);
            note_slack(hit, amargin);
            disclose_workspace(hit, step, "payload@");
            disclose_provider(hit, step, "payload@");
            if (hit.payload_own_pairs > 0) {
              const auto wall = std::chrono::steady_clock::now();
              if (wall - payload_own_last_warn_ >= std::chrono::seconds(1)) {
                payload_own_last_warn_ = wall;
                RCLCPP_WARN(this->get_logger(),
                            "safety.payload_own_cells step=%d pairs=%d deepest=%.1f mm: the "
                            "payload's own measured cells tolerated (never a stop's evidence)",
                            step, hit.payload_own_pairs, hit.payload_own_min_distance * 1000.0);
              }
            }
            if (hit.hit) {
              // ADR-0098: when declared target geometry adjudicated the pair,
              // hit.min_distance is the distance to that BODY, not the cell
              // link_b indexes, so the evidence must name the body —
              // quoting one geometry's distance under another's identity is
              // the CollisionHit-forbidden failure #187 fixed.
              report("world", WsName{'a', hit.link_a},
                     WsName{hit.place_target_adjudicated ? 'p' : 'v', hit.link_b}, step, hit);
              return true;
            }
          }
          const auto hit = check_attached_self_collision(
              collision_model_, attached_model_, collision_scratch_, amargin,
              workspace_contact_enabled_ ? &voxel_grid_.workspace_region : nullptr, provider_gate);
          note_slack(hit, amargin);
          disclose_workspace(hit, step, "payload-self@");
          disclose_provider(hit, step, "payload-self@");
          if (hit.hit) {
            report("self", WsName{'a', hit.link_a}, WsName{'l', hit.link_b}, step, hit);
            return true;
          }
          // ADR-0100 contact-force gate, survey Path C. Evaluated LAST, after
          // every geometric check above passed — makes "additive
          // conservatism" literally true: the only configurations this can
          // reach are ones the kernel was about to accept, so it can only
          // turn an accept into a refusal, never widen a margin or create an
          // exemption; geometry keeps priority in the evidence.
          //
          // Configuration-invariant across the horizon (witness describes
          // measured contact now, not a predicted one), so it's reported at
          // the step being checked and the whole chunk is refused, exactly
          // as any other hit at that step would be.
          const auto force = check_contact_force_gate(attached_model_, this->now().nanoseconds(),
                                                      envelope_.contact_force_threshold_n);
          if (force.tripped) {
            CollisionHit fhit{};
            fhit.hit = true;
            fhit.link_a = force.object_index;
            fhit.link_b = -1;
            // There is no geometric pair here, and inventing a distance for one
            // would put a number in the evidence that no check measured — the
            // single-pair-coherence rule `CollisionHit` exists to enforce. The
            // force numbers travel in the log line below instead.
            fhit.min_distance = std::numeric_limits<double>::infinity();
            fhit.sweep_min_distance = std::numeric_limits<double>::infinity();
            RCLCPP_ERROR(this->get_logger(),
                         "safety.contact_force_gate object=%s target=%s magnitude_n=%g "
                         "threshold_n=%g step=%d",
                         attached_label(force.object_index).c_str(),
                         place_declaration_target_.c_str(), force.magnitude_n, force.threshold_n,
                         step);
            report("world", WsName{'a', force.object_index}, WsName{'f', 0}, step, fhit);
            return true;
          }
        }
        return false;
      };

      if (is_position) {
        // #349 hand speed and lead. The speed bound is about impact energy. Two
        // readings, the larger governs (#349 review B): the measured hand speed,
        // FK(q_meas + dq_meas * period) against FK(q_meas) over one period, the
        // same for every row; and the commanded row-to-row speed below, so a fast
        // command issued from rest does not read as slow. The chunk executor's
        // 0.5-1 rad chunk-boundary steps read fast this way: under the brake such
        // a speed-only violation creeps at the bounded speed instead of stopping.
        //
        // Commanded speed: row s is reached from row s-1 in one
        // control period. Row 0's previous row is, per joint the row commands,
        // that joint's previously ACCEPTED command while it is fresh, else its
        // measured value (a position-controlled arm trails its command by
        // several periods, so measured-to-row reads tracking lag as speed;
        // the fallback can only read the hand faster, never slower). Joints the
        // row does not command keep the value the check places them at (the
        // measured state; for the tick-partner check, the partner's committed
        // target), so they add no phantom motion. The lead is always taken
        // against the MEASURED configuration.
        std::vector<Transform>* ws_plain = nullptr;
        std::vector<Transform>* ws_partner = nullptr;
        if (workspace_speed_known) {
          const std::int64_t now_ns = this->now().nanoseconds();
          const bool full_row = msg->joint_names.empty();
          fk_config(q_meas_.data());
          workspace_snapshot(workspace_meas_pose_);
          // Is the measured hand (or its payload) already touching? Every hand pair
          // read as tolerated (speed and lead 0), so only contact is asked; an arm
          // link's hit counts too (#349 review A: a touching hand gets no per-row
          // lead allowance).
          if (world_voxel_enabled_) {
            voxel_grid_.workspace_region.hand_speed_mps = 0.0;
            voxel_grid_.workspace_region.hand_lead_m = 0.0;
            const auto touch = check_voxel_collision(collision_model_, collision_scratch_,
                                                     voxel_grid_, world_voxel_margin_m_);
            ws_hand_contact_ = touch.hit || touch.workspace_pairs > 0;
            if (!ws_hand_contact_ && attached_collision_enabled_ && attached_model_.n_objects > 0) {
              // The own-cell exemption is a payload-vs-voxel relaxation only; whether the hand
              // touches anything is measured without it.
              const PayloadOwnCells* const saved_own = voxel_grid_.payload_own;
              voxel_grid_.payload_own = nullptr;
              const auto ptouch = check_attached_voxel_collision(collision_model_, attached_model_,
                                                                 collision_scratch_, voxel_grid_,
                                                                 attached_collision_margin_m_);
              voxel_grid_.payload_own = saved_own;
              ws_hand_contact_ = ptouch.hit || ptouch.workspace_pairs > 0;
            }
          }
          if (measured_velocity_fresh(now_ns)) {
            workspace_snapshot(workspace_prev_plain_);
            std::copy(q_meas_.begin(), q_meas_.end(), q_workspace_.begin());
            for (const int d : collision_fk_dofs_) {
              // The jaws' own velocity is not the hand's: a jaw squeezing an object
              // jitters at its contact (#349 p4_mix00, 0.11 -> 0.38 m/s in a second
              // with the arm slow); its closing force is the gripper's effort limit.
              if (std::find(workspace_jaw_cols_.begin(), workspace_jaw_cols_.end(), d) !=
                  workspace_jaw_cols_.end()) {
                continue;
              }
              const auto i = static_cast<std::size_t>(d);
              q_workspace_[i] += dq_meas_[i] * workspace_control_period_s_;
            }
            fk_config(q_workspace_.data());
            workspace_measured_speed_mps_ = workspace_hand_speed(workspace_prev_plain_);
            workspace_speed_measured_ = true;
          }
          for (std::size_t d = 0; d < robot_ndof; ++d) {
            const bool commanded = full_row || slot_commanded_[d] != 0;
            q_workspace_[d] =
                commanded && workspace_command_fresh(d, now_ns) ? workspace_cmd_q_[d] : q_meas_[d];
          }
          fk_config(q_workspace_.data());
          workspace_snapshot(workspace_prev_plain_);
          ws_plain = &workspace_prev_plain_;
          if (tick_partner) {
            for (std::size_t d = 0; d < robot_ndof; ++d) {
              if (slot_commanded_[d] == 0 && tick_target_set_[d] != 0) {
                q_workspace_[d] = tick_target_[d];
              }
            }
            fk_config(q_workspace_.data());
            workspace_snapshot(workspace_prev_partner_);
            ws_partner = &workspace_prev_partner_;
          }
        }
        // A slot row in `q_check_` (its uncommanded joints measured) re-checked at this
        // tick's other committed targets. Only ever an extra check: it can refuse, never
        // accept. Overwrites `q_check_`'s uncommanded joints.
        const auto partner_trips = [&](int step) -> bool {
          if (!tick_partner) {
            return false;
          }
          for (std::size_t d = 0; d < robot_ndof; ++d) {
            if (slot_commanded_[d] == 0 && tick_target_set_[d] != 0) {
              q_check_[d] = tick_target_[d];
            }
          }
          return check_config(q_check_.data(), step, 0.0, ws_partner);
        };
        const auto check_rows = [&]() -> bool {
          for (std::uint16_t s = 0; s < view.horizon; ++s) {
            const double* row = view.flat_data + static_cast<std::size_t>(s) * robot_ndof;
            if (!slot_row) {
              // A full row is the joint configuration itself.
              if (check_config(row, static_cast<int>(s), 0.0, ws_plain)) {
                return true;
              }
              continue;
            }
            // Slot row: commanded joints from the row, the rest measured ...
            for (std::size_t d = 0; d < robot_ndof; ++d) {
              q_check_[d] = slot_commanded_[d] != 0 ? row[d] : q_meas_[d];
            }
            if (check_config(q_check_.data(), static_cast<int>(s), 0.0, ws_plain)) {
              return true;
            }
            // ... and, additionally, at this tick's other committed targets.
            if (partner_trips(static_cast<int>(s))) {
              return true;
            }
          }
          return false;
        };
        // #349 workspace brake (sim-only draft). With a live workspace and a fresh
        // measured state the rows are first checked as a dry run (nothing reported,
        // disclosure kept): no trip is today's accept, at today's cost. On a trip
        // they are checked again with the workspace hand speed AND lead read as
        // within bound — every other check, every pair's exemption and the measured
        // pose itself unchanged. Passing then means the speed and/or the push were
        // the ONLY reasons to stop, and a hold at the measured pose answers both:
        // no further motion and no push (lead 0) is commanded, power is kept, no
        // latch, and the next chunk is judged afresh. Jaw joints keep their command
        // (a hold must not drop the payload). Any other reason stops as before.
        bool rows_checked = false;
        if (workspace_brake_enabled_ && voxel_grid_.workspace_region.valid &&
            workspace_speed_measured_) {
          // Holds are counted per commanded joint (#349 review): with slot dispatch the
          // other arm's chunk passing must not reset this arm's run of holds.
          const bool full_cmd = msg->joint_names.empty();
          const auto commanded_dof = [&](std::size_t d) {
            return d < workspace_command_holds_.size() && (full_cmd || slot_commanded_[d] != 0);
          };
          const bool full_row = full_cmd;
          const auto is_jaw = [&](std::size_t d) {
            return std::find(workspace_jaw_cols_.begin(), workspace_jaw_cols_.end(),
                             static_cast<int>(d)) != workspace_jaw_cols_.end();
          };
          std::uint64_t holds = 0;
          for (std::size_t d = 0; d < robot_ndof; ++d) {
            if (commanded_dof(d)) {
              holds = std::max(holds, workspace_command_holds_[d]);
            }
          }
          ws_dry_run_ = true;
          ws_first_trip_.set = false;
          const bool as_is = check_rows();
          ws_dry_run_ = false;
          rows_checked = !as_is;
          if (!as_is) {
            for (std::size_t d = 0; d < robot_ndof; ++d) {
              if (commanded_dof(d)) {
                workspace_command_holds_[d] = 0;
              }
            }
          }
          // The brake decision for a refused chunk: dry, quiet passes only — no report,
          // no log, no message, no allocation (#349 review F; `workspace_brake_probe`
          // brackets it for the allocation test). Its outcome is acted on below.
          bool rows_refused = false;
          bool hold_trips = true;
          bool slowed = true;
          bool hold_clear = false;
          double creep = 0.0;
          if (as_is) {
            if (const auto probe = workspace_brake_probe.load(std::memory_order_relaxed)) {
              probe(true);
            }
            ws_dry_run_ = true;
            ws_quiet_ = true;
            ws_speed_override_ = true;
            ws_refused_trip_.set = false;
            rows_refused = check_rows();
            // The hold IS the measured pose: it must pass.
            std::copy(q_meas_.begin(), q_meas_.end(), q_check_.begin());
            hold_trips = check_config(q_check_.data(), -1, 0.0, ws_plain);
            // Speed/lead only (rows pass relaxed): always a hold. A command the relaxed
            // check still refuses: a hold up to the consecutive cap, then a stop.
            const bool command_hold = rows_refused && holds < workspace_hold_max_consecutive_;
            slowed = hold_trips || (rows_refused && !command_hold);
            ws_creep_lead_m_ = 0.0;
            ws_creep_row_ = -1;
            if (!slowed && !rows_refused && view.horizon > 0) {
              // Creep, not freeze, when speed/lead are the only reasons (#349 r24/r25: a
              // held policy re-issues the same command and stalls, 200-1700 holds a run):
              // the commanded rows scaled toward the measured pose so the tightest row
              // leads by one control period at the speed bound while the measured hand is
              // over its speed bound, else by half the lead bound (r32: scaling a 0.12 m
              // lead to one period, 0.017 m, stalled the policy as a hold did); row s adds
              // its allowance (s periods at the bound; none while the hand touches, #349
              // review A). The scaled rows must pass the relaxed check themselves;
              // otherwise (and for a refused command) the measured pose is held.
              const double per_period = workspace_max_speed_mps_ * workspace_control_period_s_;
              const double base = workspace_measured_speed_mps_ > workspace_max_speed_mps_
                                      ? std::min(0.5 * workspace_max_lead_m_, per_period)
                                      : 0.5 * workspace_max_lead_m_;
              const double per_row = ws_hand_contact_ ? 0.0 : per_period;
              // The same speed rule as the row check: row 0 is bounded by the lead (its
              // step is the chunk-boundary jump), rows s >= 1 also by the row-to-row speed
              // — scaling toward the measured pose scales those steps by `creep`.
              creep = 1.0;
              for (std::uint16_t s = 0; s < view.horizon; ++s) {
                const double* row = view.flat_data + static_cast<std::size_t>(s) * robot_ndof;
                for (std::size_t d = 0; d < robot_ndof; ++d) {
                  const bool commanded = full_row || slot_commanded_[d] != 0;
                  q_check_[d] = commanded && !is_jaw(d) ? row[d] : q_meas_[d];
                }
                fk_config(q_check_.data());
                const double lead = workspace_hand_lead();
                const double allow = base + per_row * static_cast<double>(s);
                if (std::isfinite(lead) && lead > allow && allow / lead < creep) {
                  creep = allow / lead;
                  ws_creep_lead_m_ = lead;
                  ws_creep_row_ = static_cast<int>(s);
                }
                if (s == 0) {
                  workspace_snapshot(workspace_creep_prev_);
                } else {
                  const double v = workspace_hand_speed(workspace_creep_prev_);
                  if (v > workspace_max_speed_mps_ && workspace_max_speed_mps_ / v < creep) {
                    creep = workspace_max_speed_mps_ / v;
                    ws_creep_row_ = static_cast<int>(s);
                  }
                }
              }
              // Lead and speed are Cartesian, not linear in the joint deltas (#349 review):
              // every scaled row must pass its own lead, speed and relaxed check. On an
              // overshoot the creep halves and is re-checked, so a creep still moves
              // (#349 r41: creep=0.000 held the arm still); after three halvings, a hold.
              // ponytail: halving, not a solve; a bisection if holds stay common.
              bool clear = false;
              for (int attempt = 0; attempt < 4 && creep > 0.0 && !clear; ++attempt) {
                if (attempt > 0) {
                  creep *= 0.5;
                }
                clear = true;
                for (std::uint16_t s = 0; s < view.horizon && clear; ++s) {
                  const double* row = view.flat_data + static_cast<std::size_t>(s) * robot_ndof;
                  for (std::size_t d = 0; d < robot_ndof; ++d) {
                    const bool commanded = full_row || slot_commanded_[d] != 0;
                    q_check_[d] = !commanded  ? q_meas_[d]
                                  : is_jaw(d) ? row[d]
                                              : q_meas_[d] + creep * (row[d] - q_meas_[d]);
                  }
                  std::copy(q_check_.begin(), q_check_.end(), q_lead_.begin());
                  for (const int jd : workspace_jaw_cols_) {
                    q_lead_[static_cast<std::size_t>(jd)] = q_meas_[static_cast<std::size_t>(jd)];
                  }
                  fk_config(q_lead_.data());
                  const double scaled_lead = workspace_hand_lead();
                  double v = 0.0;
                  if (s == 0) {
                    workspace_snapshot(workspace_creep_prev_);
                  } else {
                    v = workspace_hand_speed(workspace_creep_prev_);
                  }
                  clear = scaled_lead <= base + per_row * static_cast<double>(s) + 1e-9 &&
                          v <= workspace_max_speed_mps_ + 1e-9 &&
                          !check_config(q_check_.data(), static_cast<int>(s), 0.0, ws_plain) &&
                          !partner_trips(static_cast<int>(s));
                }
              }
              if (!clear) {
                creep = 0.0;  // the scaled motion itself is not clear: hold
              }
            }
            if (!slowed) {
              // A hold publishes the arm at measured but the jaws at their command (a hold
              // must not drop the payload): every row of exactly that must pass (#349
              // review: only the all-measured pose was checked, so a refused jaw motion
              // was executed under the hold).
              hold_clear = true;
              if (creep == 0.0 && !workspace_jaw_cols_.empty()) {
                ws_refused_trip_.set = false;
                for (std::uint16_t s = 0; s < view.horizon && hold_clear; ++s) {
                  const double* row = view.flat_data + static_cast<std::size_t>(s) * robot_ndof;
                  for (std::size_t d = 0; d < robot_ndof; ++d) {
                    const bool commanded = full_row || slot_commanded_[d] != 0;
                    q_check_[d] = commanded && is_jaw(d) ? row[d] : q_meas_[d];
                  }
                  hold_clear = !check_config(q_check_.data(), static_cast<int>(s), 0.0, ws_plain) &&
                               !partner_trips(static_cast<int>(s));
                }
              }
            }
            ws_speed_override_ = false;
            ws_quiet_ = false;
            ws_dry_run_ = false;
            if (const auto probe = workspace_brake_probe.load(std::memory_order_relaxed)) {
              probe(false);
            }
          }
          if (as_is && !slowed && hold_clear) {
            if (rows_refused) {
              for (std::size_t d = 0; d < robot_ndof; ++d) {
                if (commanded_dof(d)) {
                  ++workspace_command_holds_[d];
                }
              }
            }
            scaled_chunk_ = *msg;
            for (std::uint16_t s = 0; s < view.horizon; ++s) {
              for (std::size_t d = 0; d < robot_ndof; ++d) {
                const std::size_t k = static_cast<std::size_t>(s) * robot_ndof + d;
                if ((full_row || slot_commanded_[d] != 0) && !is_jaw(d) &&
                    k < scaled_chunk_.flat.size()) {
                  scaled_chunk_.flat[k] = q_meas_[d] + creep * (scaled_chunk_.flat[k] - q_meas_[d]);
                }
              }
            }
            safe_pub_->publish(scaled_chunk_);
            ++workspace_brakes_;
            // A brake is an accept of the scaled chunk: the status says so, and the
            // tick's later slots are checked against what this slot now commits
            // (#349 review: they were checked against its measured pose).
            set_safety_status(false, openral_msgs::msg::SafetyStatus::DROP_NONE, "workspace brake",
                              msg->rskill_id, msg->trace_id);
            if (slot_row && msg->tick_index != 0 && view.horizon > 0) {
              const std::size_t last = static_cast<std::size_t>(view.horizon - 1) * robot_ndof;
              for (std::size_t d = 0; d < robot_ndof; ++d) {
                if (slot_commanded_[d] != 0 && last + d < scaled_chunk_.flat.size()) {
                  tick_target_[d] = scaled_chunk_.flat[last + d];
                  tick_target_set_[d] = 1;
                }
              }
            }
            clear_workspace_commands();
            const auto wall = std::chrono::steady_clock::now();
            if (wall - workspace_brake_last_warn_ >= std::chrono::seconds(1)) {
              workspace_brake_last_warn_ = wall;
              RCLCPP_WARN(this->get_logger(),
                          "safety.workspace_brake hand_speed_mps=%.3f max_mps=%g max_lead_m=%g "
                          "brakes=%llu reason=%s refused=%s creep=%.3f lead_m=%.3f row=%d "
                          "target=%s rskill_id=%s (0 = held at the measured pose; no latch)",
                          workspace_measured_speed_mps_, workspace_max_speed_mps_,
                          workspace_max_lead_m_, static_cast<unsigned long long>(workspace_brakes_),
                          rows_refused ? "command_refused" : "speed_lead",
                          rows_refused ? trip_str(ws_refused_trip_).c_str() : "", creep,
                          ws_creep_lead_m_, ws_creep_row_, workspace_target_.c_str(),
                          msg->rskill_id.c_str());
            }
            span->SetAttribute("safety.severity", "info");
            span->SetAttribute("safety.workspace_brake", true);
            span->SetAttribute("safety.workspace_hand_speed_mps", workspace_measured_speed_mps_);
            span->End();
            return;
          }
          if (as_is) {
            if (slowed) {
              RCLCPP_WARN_THROTTLE(
                  this->get_logger(), *this->get_clock(), 1000,
                  "safety.workspace_brake_refused hand_speed_mps=%.3f pair=%s "
                  "hold_clear=%d command_holds=%llu/%llu (stops as before)",
                  workspace_measured_speed_mps_, trip_str(ws_refused_trip_).c_str(),
                  static_cast<int>(!hold_trips), static_cast<unsigned long long>(holds),
                  static_cast<unsigned long long>(workspace_hold_max_consecutive_));
            } else {
              RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                   "safety.workspace_brake_refused hand_speed_mps=%.3f pair=%s "
                                   "jaw_hold_clear=0 (stops as before)",
                                   workspace_measured_speed_mps_,
                                   trip_str(ws_refused_trip_).c_str());
            }
            // Stops as before, on the dry run's own first trip: reported as recorded,
            // at the configuration and hand readings it was measured at, not re-checked
            // (#349 review F: no re-run, no second disclosure).
            const WsTrip t = ws_first_trip_;
            std::copy(ws_trip_q_.begin(), ws_trip_q_.end(), q_fk_.begin());
            voxel_grid_.workspace_region.hand_speed_mps = t.hand_speed_mps;
            voxel_grid_.workspace_region.hand_lead_m = t.hand_lead_m;
            ws_speed_measured_row_ = t.speed_measured;
            report(t.kind, t.a, t.b, t.step, t.hit);
            return;
          }
        }
        if (!rows_checked && check_rows()) {
          return;
        }
        // #349: accepted — remember each commanded joint's final row as the
        // previous row of the next chunk's row 0, due once the chunk has
        // played out ((horizon-1) periods from now).
        if (workspace_contact_enabled_ && view.horizon > 0) {
          const double* last =
              view.flat_data + static_cast<std::size_t>(view.horizon - 1) * robot_ndof;
          const std::int64_t due_ns =
              this->now().nanoseconds() +
              static_cast<std::int64_t>(static_cast<double>(view.horizon - 1) *
                                        workspace_control_period_s_ * 1e9);
          const bool full_row = msg->joint_names.empty();
          for (std::size_t d = 0; d < robot_ndof && d < workspace_cmd_q_.size(); ++d) {
            if (full_row || slot_commanded_[d] != 0) {
              workspace_cmd_q_[d] = last[d];
              workspace_cmd_due_ns_[d] = due_ns;
              workspace_cmd_set_[d] = 1;
            }
          }
        }
        // ADR-0121: tell the provider which command was accepted (its chain's next row 0).
        if (provider_ != nullptr && view.horizon > 0) {
          const bool full_row = msg->joint_names.empty();
          CandidateOutcome outcome;
          outcome.q_last_row =
              view.flat_data + static_cast<std::size_t>(view.horizon - 1) * robot_ndof;
          outcome.commanded = full_row ? nullptr : slot_commanded_.data();
          outcome.n_dof = robot_ndof;
          outcome.horizon = view.horizon;
          outcome.accepted_ns = this->now().nanoseconds();
          provider_->end_candidate(outcome);
        }
        // Accepted: this slot's final target is what the tick's later slots
        // must be checked against.
        if (slot_row && msg->tick_index != 0 && view.horizon > 0) {
          const double* last =
              view.flat_data + static_cast<std::size_t>(view.horizon - 1) * robot_ndof;
          for (std::size_t d = 0; d < robot_ndof; ++d) {
            if (slot_commanded_[d] != 0) {
              tick_target_[d] = last[d];
              tick_target_set_[d] = 1;
            }
          }
        }
      } else {  // is_velocity (Phase 2) or is_cartesian (Phase 3)
        // Reactive (both modes): the current measured configuration itself —
        // catches an arm already in/at an obstacle. This is what covers the
        // Cartesian-delta arm chunk (LIBERO/SIMPLER/DROID + the robocasa arm).
        std::copy(q_meas_.begin(), q_meas_.end(), q_check_.begin());
        if (check_config(q_check_.data(), -1)) {  // step -1 = measured state
          return;
        }
        // Predictive (velocity only): integrate the commanded joint velocities
        // forward (q_s = q_meas + Σ_{i≤s} v_i·dt) and check each step so a command
        // that *would* drive into an obstacle is rejected before it is applied.
        // dt=0 keeps reactive-only.
        if (is_velocity && rows_full_dof && collision_seed_dt_s_ > 0.0) {
          for (std::uint16_t s = 0; s < view.horizon; ++s) {
            const double* row = view.flat_data + static_cast<std::size_t>(s) * robot_ndof;
            for (std::size_t d = 0; d < robot_ndof; ++d) {
              q_check_[d] += row[d] * collision_seed_dt_s_;
            }
            if (check_config(q_check_.data(), static_cast<int>(s))) {
              return;
            }
          }
        }
        // Predictive Cartesian (CARTESIAN_DELTA, Phase 3): reconstruct
        // where the proposed EE deltas drive the ARM via the damped-least-squares
        // Jacobian and check the full capsule boundary at each look-ahead step.
        // The user contract: at minimum the LAST action in the chunk is verified
        // safe; intermediate steps are checked up to the budget. The reactive
        // check above is the guaranteed floor, so an imperfect IK reconstruction
        // can only ADD early rejections — never make the kernel less safe.
        // Rows are EE twists (base frame assumed) with stride view.n_dof; the
        // first 6 entries are [vx,vy,vz, wx,wy,wz]. Disabled (ee_link<0) for
        // robots whose EE link is not plumbed, and skipped for non-delta modes
        // (CARTESIAN_POSE/TWIST/BODY_TWIST stay reactive — fail-safe).
        if (is_cartesian && mode == ControlMode::kCartesianDelta && collision_ee_link_ >= 0 &&
            view.n_dof >= 6 && view.flat_data != nullptr) {
          std::copy(q_meas_.begin(), q_meas_.end(), q_predict_.begin());
          for (const int d : collision_base_dofs_) {
            q_predict_[static_cast<std::size_t>(d)] = 0.0;
          }
          const std::size_t cap =
              collision_predict_max_steps_ == 0 ? view.horizon : collision_predict_max_steps_;
          for (std::uint16_t s = 0; s < view.horizon; ++s) {
            // FK the current predicted config (base-relative) so the Jacobian is
            // taken at q_predict; integrate q_predict every step to keep the
            // trajectory correct even when a step's check is skipped by the cap.
            fk_config(q_predict_.data());
            const double* row = view.flat_data + static_cast<std::size_t>(s) * view.n_dof;
            const auto physical_delta_at = [&](std::size_t i) {
              if (view.cartesian_delta_scale_size == 0) {
                return row[i];  // identity: policy already emits physical units
              }
              // Native normalized OSC controllers clip to [-1, 1] before
              // multiplying by their per-axis output range.
              return std::clamp(row[i], -1.0, 1.0) * view.cartesian_delta_scale[i];
            };
            const double twist[6] = {physical_delta_at(0), physical_delta_at(1),
                                     physical_delta_at(2), physical_delta_at(3),
                                     physical_delta_at(4), physical_delta_at(5)};
            if (!jacobian_dls_step(collision_model_, collision_scratch_, collision_ee_link_, twist,
                                   collision_predict_lambda_, dq_.data(), robot_ndof,
                                   dof_blocked_.data())) {
              break;  // cannot reconstruct (singular/blocked) → reactive floor stands
            }
            for (std::size_t d = 0; d < robot_ndof; ++d) {
              q_predict_[d] += dq_[d];
            }
            // Always check the final step; earlier steps up to the budget.
            const bool last = (static_cast<std::size_t>(s) + 1 == view.horizon);
            if (s < cap || last) {
              const double extra = collision_predict_margin_growth_m_ * static_cast<double>(s);
              if (check_config(q_predict_.data(), static_cast<int>(s), extra)) {
                return;
              }
            }
          }
        }
      }
    }

    // ADR-0096 recovery transition — a chunk that cleared every check ends
    // whatever non-latching fail-closed drop was in effect, so the latched
    // status must say so rather than staying on the last drop reason
    // forever. Cost on the pass-through hot path when nothing changed (the
    // overwhelmingly common case) is two integer comparisons inside
    // set_safety_status, no string touched, no publish, no allocation.
    set_safety_status(false, openral_msgs::msg::SafetyStatus::DROP_NONE, "chunk accepted",
                      msg->rskill_id, msg->trace_id);

    // Distance-graded slowdown (#188). Chunk is ALREADY accepted — this
    // never turns an accept into a drop or vice versa; it only decides how
    // fast the accepted motion executes.
    //
    // Only rate-shaped modes are scalable: multiplying an ABSOLUTE target
    // (JOINT_POSITION, CARTESIAN_POSE, JOINT_TRAJECTORY) by 0.5 doesn't halve
    // the speed, it commands a pose halfway to the origin — a large
    // unrequested motion in an unpredictable direction. Shrinking toward the
    // measured configuration would be correct but needs a fresh seed
    // position mode doesn't require today, so absolute modes are left alone.
    // Gripper/dex-hand/composite rows aren't motion rates either (a scaled
    // grasp drops the object; a scaled mode flag corrupts the multiplexer).
    // BODY_TWIST is excluded on evidence: FK zeroes the base dofs, so
    // nothing here measures where the BASE is going, and slowing it for an
    // arm-side clearance would apply the number to the wrong body (base's
    // own proximity gate is Nav2's collision monitor, #186).
    const bool rate_shaped = mode == ControlMode::kJointVelocity ||
                             mode == ControlMode::kCartesianDelta ||
                             mode == ControlMode::kCartesianTwist;
    const double scale =
        (geom_enabled && rate_shaped) ? velocity_scale_for(collision_min_slack_m) : 1.0;
    if (scale < 1.0) {
      // Scaling DOWN a rate keeps every envelope bound it already satisfied
      // (|s·v| <= |v| for s in [0,1]), so the scaled chunk needs no
      // re-validation, and it travels no further than the swept volume the
      // predictive check just cleared. Strictly more conservative, both ways.
      scaled_chunk_ = *msg;
      const std::size_t stride = view.n_dof;
      // A Cartesian row is a 6-vector twist [vx,vy,vz,wx,wy,wz] inside a row of
      // `n_dof`; a velocity row is velocities all the way across.
      const std::size_t scalable =
          (mode == ControlMode::kJointVelocity) ? stride : std::min<std::size_t>(stride, 6);
      // A NORMALIZED Cartesian chunk must be clamped before it is scaled.
      // Native OSC controllers apply `clamp(raw, -1, 1) * per_axis_range`, and
      // the validator deliberately puts no per-axis bound on CARTESIAN_DELTA,
      // so |raw| > 1 is admissible and common. Multiplying 2.5 by 0.5 gives
      // 1.25, which still clips to 1.0 downstream — the identical motion, while
      // the kernel logs and counts a slowdown that did not happen. Clamping
      // first makes the number the robot will actually use the number that gets
      // scaled, so the disclosure describes a real effect.
      const bool normalized = view.cartesian_delta_scale_size != 0;
      for (std::size_t s = 0; s < view.horizon && stride > 0; ++s) {
        const std::size_t row = s * stride;
        for (std::size_t d = 0; d < scalable && row + d < scaled_chunk_.flat.size(); ++d) {
          double& value = scaled_chunk_.flat[row + d];
          if (normalized) {
            value = std::clamp(value, -1.0, 1.0);
          }
          value *= scale;
        }
      }
      safe_pub_->publish(scaled_chunk_);
      ++chunks_scaled_;
      // Transition-gated, not per chunk. This is the accept path, which runs at
      // 30-200 Hz and whose pass-through case is documented as touching no
      // string and making no publish; an armed band near an obstacle would
      // otherwise format two std::strings every tick inside a node whose
      // realtime rules forbid string ops in hot loops. A tenth of a scale is
      // the granularity worth a line; the 1 Hz /diagnostics `scaled` counter
      // covers the "has been crawling for a minute" case continuously.
      if (std::abs(scale - last_logged_scale_) >= 0.1) {
        last_logged_scale_ = scale;
        RCLCPP_INFO(this->get_logger(),
                    "safety.collision_scaled scale=%g slack_m=%g band_m=%g mode=%u rskill_id=%s",
                    scale, collision_min_slack_m, collision_scale_proximity_m_,
                    static_cast<unsigned>(view.control_mode), msg->rskill_id.c_str());
      }
      span->SetAttribute("safety.velocity_scale", scale);
      span->SetAttribute("safety.scale_slack_m", collision_min_slack_m);
    } else {
      // Leaving the band is a transition worth one line too, so a reader can
      // see the slowdown end rather than infer it from silence.
      if (last_logged_scale_ < 1.0) {
        last_logged_scale_ = 1.0;
        RCLCPP_INFO(this->get_logger(), "safety.collision_scaled scale=1 (cleared the band)");
      }
      safe_pub_->publish(*msg);
    }
    ++chunks_passed_;
    // An accepted chunk is what "the payload backed off" looks like from here,
    // so the advisory run (#176) starts again. Only an unbroken run of refusals
    // counts toward the cap; a skill that alternates progress with the odd
    // in-receptacle graze is making progress, not shoving.
    advisory_refusals_ = 0;
    span->SetAttribute("safety.severity", "info");
    span->End();
    return;
  }

  // Violation: drop + publish failure + fire estop + latch.
  const Violation& v = result.error();
  ++chunks_dropped_;
  last_drop_reason_ = violation_kind_field(v.kind);
  fault_latch_ = true;
  last_estop_at_ = std::chrono::steady_clock::now();

  publish_failure_trigger(*msg, v);
  // ADR-0096 — same fault, now also as durable current state. The numeric
  // drop_reason is the FailureTrigger KIND_* the trigger above carries, so
  // one number means the same fault on both topics.
  set_safety_status(true, violation_kind_constant(v.kind), v.field, msg->rskill_id, msg->trace_id);

  std_msgs::msg::Empty estop_msg;
  estop_pub_->publish(estop_msg);

  RCLCPP_ERROR(this->get_logger(),
               "safety.envelope_violation kind=%s field=%s joint=%u step=%u value=%g limit=%g "
               "rskill_id=%s trace_id=%s",
               violation_kind_field(v.kind), v.field, static_cast<unsigned>(v.joint_index),
               static_cast<unsigned>(v.horizon_step), v.offending_value, v.limit_value,
               msg->rskill_id.c_str(), msg->trace_id.c_str());

  // Record the typed violation on the span so the dashboard's Safety
  // card surfaces the drop reason + the Event log ticks. The Python
  // SafetyPassthroughNode emits the same shape (supervisor_node.py:286).
  span->SetAttribute("safety.severity", "violation");
  span->SetAttribute("safety.drop_reason", violation_kind_field(v.kind));
  span->SetAttribute("safety.violation_reason", v.field);
  span->SetAttribute("safety.violation_joint", static_cast<int64_t>(v.joint_index));
  span->SetAttribute("safety.violation_value", v.offending_value);
  span->SetAttribute("safety.violation_limit", v.limit_value);
  span->AddEvent(otel::kSafetyViolationEventName, {
                                                      {"safety.kind", violation_kind_field(v.kind)},
                                                      {"safety.field", v.field},
                                                  });
  span->End();
}

void SafetyKernelLifecycleNode::on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg) {
  // Phase 1 — fold the measured positions into q_meas_ in the action's
  // dof order. Single-threaded executor → direct write, no lock (mirrors the
  // world/voxel ingest). Unknown joint names are ignored; missing FK-relevant
  // dofs leave q_meas_seen_ false so measured_state_fresh() fails closed.
  if (msg == nullptr) {
    return;
  }
  // #349: the measured velocity rides the same message. A joint named without
  // a finite velocity entry forgets its last one, so the workspace speed bound
  // falls back to the commanded speed rather than trusting an old reading.
  const std::int64_t now_ns = this->now().nanoseconds();
  const std::int64_t src_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
  const std::size_t n = std::min(msg->name.size(), msg->position.size());
  for (std::size_t i = 0; i < n; ++i) {
    const auto it = joint_name_to_dof_.find(msg->name[i]);
    if (it == joint_name_to_dof_.end()) {
      continue;
    }
    const std::size_t d = static_cast<std::size_t>(it->second);
    if (d < q_meas_.size()) {
      const bool has_v = i < msg->velocity.size() && std::isfinite(msg->velocity[i]);
      double v = has_v ? msg->velocity[i] : 0.0;
      // A reported velocity is capped at the rate its own measured position moved since
      // the joint's previous stamped message (Spark R262, 2026-10-22: PhysX reported
      // wrist_roll at its 10 rad/s limit while it sat 6 mrad from the command, a 1.88 m/s
      // "hand" that stopped the grasp). Unstamped, repeated or out-of-order messages are
      // not corroborated: the raw reading stands. Feeds only the #349 workspace speed.
      // Safety-WG review owed.
      if (has_v && q_meas_seen_[d] && q_meas_src_ns_[d] > 0 && src_ns > q_meas_src_ns_[d] &&
          std::isfinite(msg->position[i])) {
        const double rate = std::abs(msg->position[i] - q_meas_[d]) /
                            (static_cast<double>(src_ns - q_meas_src_ns_[d]) * 1e-9);
        v = std::clamp(v, -rate, rate);
      }
      q_meas_src_ns_[d] = src_ns > 0 ? src_ns : 0;
      if (force_observer_enabled_) {
        // `effort=[]` means "not measured" (#388), never a row of zeros.
        dq_raw_[d] = has_v ? msg->velocity[i] : 0.0;
        if (i < msg->effort.size() && std::isfinite(msg->effort[i])) {
          tau_meas_[d] = msg->effort[i];
          tau_meas_stamp_ns_[d] = now_ns;
        }
      }
      q_meas_[d] = msg->position[i];
      q_meas_seen_[d] = true;
      dq_meas_[d] = v;
      dq_meas_seen_[d] = has_v ? 1 : 0;
      dq_meas_stamp_ns_[d] = now_ns;
    }
  }
  q_meas_received_ = true;
  q_meas_stamp_ = this->now();
  if (force_observer_enabled_) {
    run_force_observer(src_ns > 0 ? src_ns : now_ns);
  }
}

bool SafetyKernelLifecycleNode::measured_velocity_fresh(std::int64_t now_ns) const noexcept {
  const double deadline_ns = collision_state_deadline_s_ * 1e9;
  for (const int d : collision_fk_dofs_) {
    const auto i = static_cast<std::size_t>(d);
    if (i >= dq_meas_seen_.size() || dq_meas_seen_[i] == 0) {
      return false;
    }
    const std::int64_t age_ns = now_ns - dq_meas_stamp_ns_[i];
    if (age_ns < 0 || static_cast<double>(age_ns) > deadline_ns) {
      return false;
    }
  }
  return !collision_fk_dofs_.empty();
}

bool SafetyKernelLifecycleNode::measured_state_fresh() const noexcept {
  if (!q_meas_received_) {
    return false;
  }
  if ((this->now() - q_meas_stamp_).seconds() > collision_state_deadline_s_) {
    return false;
  }
  for (const int d : collision_fk_dofs_) {
    if (static_cast<std::size_t>(d) >= q_meas_seen_.size() ||
        !q_meas_seen_[static_cast<std::size_t>(d)]) {
      return false;
    }
  }
  return true;
}

bool SafetyKernelLifecycleNode::load_force_observer(std::string& error) {
  // Reset to the disabled state so a re-configure can never leak a stale model.
  force_observer_enabled_ = false;
  force_model_ = ForceObserverModel{};
  force_state_ = ForceObserverState{};
  force_estimate_valid_ = false;
  force_last_sample_ns_ = 0;
  force_peak_ratio_ = 0.0;
  const std::size_t ndof = envelope_.n_dof;
  joint_external_torque_max_ = this->get_parameter("joint_external_torque_max").as_double_array();
  if (!joint_external_torque_max_.empty() && joint_external_torque_max_.size() != ndof) {
    error = "joint_external_torque_max has " + std::to_string(joint_external_torque_max_.size()) +
            " entries for n_dof=" + std::to_string(ndof);
    return false;
  }
  std::size_t n_monitored = 0;
  for (const double v : joint_external_torque_max_) {
    if (!std::isfinite(v) || v < 0.0) {
      error = "joint_external_torque_max entries must be finite and >= 0 (0 = not monitored)";
      return false;
    }
    n_monitored += v > 0.0 ? 1 : 0;
  }
  joint_external_torque_max_.resize(ndof, 0.0);
  if (!this->get_parameter("force_observer_enabled").as_bool()) {
    if (n_monitored > 0) {
      // A declared ceiling that nothing enforces is the failure mode this
      // layer exists to remove (CLAUDE.md §1.4): refuse it.
      error = "joint_external_torque_max sets a ceiling but force_observer_enabled is false, "
              "so nothing would enforce it";
      return false;
    }
    return true;
  }
  if (!self_collision_enabled_ || collision_model_.n_links == 0) {
    error = "force_observer_enabled needs the self-collision kinematic tree "
            "(self_collision_enabled with a model)";
    return false;
  }
  if (!this->get_parameter("collision_base_dofs").as_integer_array().empty()) {
    error = "force_observer_enabled with collision_base_dofs: the observer models a fixed "
            "root, so a mobile base's own dynamics would read as external torque";
    return false;
  }
  const auto names = this->get_parameter("collision_joint_names").as_string_array();
  if (names.size() < ndof) {
    error = "force_observer_enabled needs collision_joint_names for every dof";
    return false;
  }
  const std::size_t n = collision_model_.n_links;
  const auto mass = this->get_parameter("force_observer_link_mass").as_double_array();
  const auto com = this->get_parameter("force_observer_link_com").as_double_array();
  const auto inertia = this->get_parameter("force_observer_link_inertia").as_double_array();
  const auto armature = this->get_parameter("force_observer_joint_armature").as_double_array();
  const auto viscous = this->get_parameter("force_observer_joint_viscous").as_double_array();
  const auto coulomb = this->get_parameter("force_observer_joint_coulomb").as_double_array();
  const auto offset = this->get_parameter("force_observer_joint_offset").as_double_array();
  const auto gravity = this->get_parameter("force_observer_gravity").as_double_array();
  if (mass.size() != n || com.size() != 3 * n || inertia.size() != 6 * n || armature.size() != n ||
      viscous.size() != n || coulomb.size() != n || offset.size() != n || gravity.size() != 3) {
    error = "force_observer link/joint arrays must be parallel to the " + std::to_string(n) +
            " collision links (mass n, com 3n, inertia 6n, joint terms n) and gravity 3";
    return false;
  }
  const auto all_finite = [](const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
  };
  if (!all_finite(mass) || !all_finite(com) || !all_finite(inertia) || !all_finite(armature) ||
      !all_finite(viscous) || !all_finite(coulomb) || !all_finite(offset) || !all_finite(gravity)) {
    error = "force_observer model has a non-finite entry";
    return false;
  }
  force_model_.links.assign(n, LinkInertial{});
  force_model_.joints.assign(n, JointFrictionModel{});
  for (std::size_t i = 0; i < n; ++i) {
    // Inertia arrives as (ixx, iyy, izz, ixy, ixz, iyz) about the COM.
    const double* t = &inertia[6 * i];
    if (mass[i] < 0.0 || t[0] < 0.0 || t[1] < 0.0 || t[2] < 0.0 || armature[i] < 0.0 ||
        viscous[i] < 0.0 || coulomb[i] < 0.0) {
      error = "force_observer link " + std::to_string(i) +
              " has a negative mass, principal inertia, armature, viscous or Coulomb term";
      return false;
    }
    LinkInertial& body = force_model_.links[i];
    body.mass = mass[i];
    body.com = Vec3{com[3 * i], com[3 * i + 1], com[3 * i + 2]};
    const double tensor[9] = {t[0], t[3], t[4], t[3], t[1], t[5], t[4], t[5], t[2]};
    std::copy(std::begin(tensor), std::end(tensor), std::begin(body.inertia));
    force_model_.joints[i] = JointFrictionModel{armature[i], viscous[i], coulomb[i], offset[i]};
  }
  force_model_.gravity = Vec3{gravity[0], gravity[1], gravity[2]};
  force_model_.gain_per_s = this->get_parameter("force_observer_gain_per_s").as_double();
  force_model_.coulomb_velocity_rad_s =
      this->get_parameter("force_observer_coulomb_velocity_rad_s").as_double();
  force_limit_margin_rad_ = this->get_parameter("force_observer_limit_margin_rad").as_double();
  if (!(std::isfinite(force_model_.gain_per_s) && force_model_.gain_per_s > 0.0) ||
      !(std::isfinite(force_model_.coulomb_velocity_rad_s) &&
        force_model_.coulomb_velocity_rad_s > 0.0) ||
      !(std::isfinite(force_limit_margin_rad_) && force_limit_margin_rad_ >= 0.0)) {
    error = "force_observer_gain_per_s and force_observer_coulomb_velocity_rad_s must be > 0, "
            "force_observer_limit_margin_rad >= 0";
    return false;
  }
  init_force_observer(collision_model_, ndof, force_state_);
  for (std::size_t d = 0; d < ndof; ++d) {
    if (joint_external_torque_max_[d] > 0.0 && force_state_.dof_link[d] < 0) {
      error = "joint_external_torque_max monitors dof " + std::to_string(d) + " (" + names[d] +
              "), which no joint of the kinematic tree drives";
      return false;
    }
  }
  tau_meas_.assign(ndof, 0.0);
  tau_meas_stamp_ns_.assign(ndof, 0);
  dq_raw_.assign(ndof, 0.0);
  external_torque_msg_ = sensor_msgs::msg::JointState{};
  external_torque_msg_.name.assign(names.begin(),
                                   names.begin() + static_cast<std::ptrdiff_t>(ndof));
  external_torque_msg_.effort.assign(ndof, 0.0);
  force_observer_enabled_ = true;
  RCLCPP_WARN(this->get_logger(),
              "safety.force_observer armed (#373, Safety-WG review pending): %zu of %zu dofs "
              "monitored, K=%g/s, limit margin %g rad. Collision tier only: an invalid "
              "estimate stops nothing and relaxes nothing",
              n_monitored, ndof, force_model_.gain_per_s, force_limit_margin_rad_);
  return true;
}

void SafetyKernelLifecycleNode::run_force_observer(std::int64_t sample_ns) {
  const std::size_t ndof = envelope_.n_dof;
  const std::int64_t now_ns = this->now().nanoseconds();
  const double deadline_ns = collision_state_deadline_s_ * 1e9;
  const auto stale = [&](std::int64_t stamp_ns) {
    const std::int64_t age_ns = now_ns - stamp_ns;
    return stamp_ns == 0 || age_ns < 0 || static_cast<double>(age_ns) > deadline_ns;
  };
  // Validity gate. The dynamics couple every column the tree moves, so each
  // needs a fresh position and velocity; a monitored column also needs a fresh
  // effort and must sit outside its limit margin (a hard stop pushes back like
  // an external torque). Anything else resets the estimate: an invalid
  // estimate never trips and never tolerates anything — the geometric checks
  // carry on exactly as without this layer.
  bool valid = true;
  for (const int d : collision_fk_dofs_) {
    const auto i = static_cast<std::size_t>(d);
    if (!q_meas_seen_[i] || dq_meas_seen_[i] == 0 || stale(dq_meas_stamp_ns_[i])) {
      valid = false;
    }
  }
  for (std::size_t d = 0; d < ndof && valid; ++d) {
    if (joint_external_torque_max_[d] <= 0.0) {
      continue;
    }
    const bool near_limit =
        d < envelope_.joint_position_min.size() && d < envelope_.joint_position_max.size() &&
        (q_meas_[d] < envelope_.joint_position_min[d] + force_limit_margin_rad_ ||
         q_meas_[d] > envelope_.joint_position_max[d] - force_limit_margin_rad_);
    if (stale(tau_meas_stamp_ns_[d]) || near_limit) {
      valid = false;
    }
  }
  if (!valid) {
    reset_force_observer(force_state_);
    force_last_sample_ns_ = 0;
    force_estimate_valid_ = false;
    return;
  }
  double dt_s = collision_state_deadline_s_;  // seed sample: any positive value
  if (force_last_sample_ns_ != 0) {
    if (sample_ns <= force_last_sample_ns_) {
      return;  // a repeated or reordered stamp carries no new interval
    }
    dt_s = static_cast<double>(sample_ns - force_last_sample_ns_) * 1e-9;
    if (dt_s > collision_state_deadline_s_) {
      reset_force_observer(force_state_);  // a gap: Ṁ and ṗ over it mean nothing
    }
  }
  force_last_sample_ns_ = sample_ns;
  if (!force_observer_update(collision_model_, force_model_, q_meas_.data(), dq_raw_.data(),
                             tau_meas_.data(), dt_s, force_state_)) {
    force_last_sample_ns_ = 0;
    force_estimate_valid_ = false;
    return;
  }
  force_estimate_valid_ = true;
  const bool can_trip = !fault_latch_ && estop_pub_ != nullptr && estop_pub_->is_activated();
  for (std::size_t d = 0; d < ndof; ++d) {
    const double limit = joint_external_torque_max_[d];
    if (limit <= 0.0) {
      continue;
    }
    const double residual = force_state_.residual[d];
    force_peak_ratio_ = std::max(force_peak_ratio_, std::abs(residual) / limit);
    if (can_trip && std::abs(residual) > limit) {
      trip_force_limit(d, residual, limit);
      break;
    }
  }
  constexpr std::int64_t kExternalTorquePublishPeriodNs = 20'000'000;  // 50 Hz
  if (external_torque_pub_ != nullptr && external_torque_pub_->is_activated() &&
      now_ns - force_last_publish_ns_ >= kExternalTorquePublishPeriodNs) {
    force_last_publish_ns_ = now_ns;
    external_torque_msg_.header.stamp =
        rclcpp::Time(sample_ns, this->get_clock()->get_clock_type());
    std::copy(force_state_.residual.begin(), force_state_.residual.end(),
              external_torque_msg_.effort.begin());
    external_torque_pub_->publish(external_torque_msg_);
  }
}

void SafetyKernelLifecycleNode::trip_force_limit(std::size_t dof, double residual_nm,
                                                 double limit_nm) {
  const std::string& joint = external_torque_msg_.name[dof];
  last_drop_reason_ = "force";
  fault_latch_ = true;
  last_estop_at_ = std::chrono::steady_clock::now();
  clear_workspace_commands(ResetReason::kLatch);

  openral_msgs::msg::FailureTrigger trigger;
  trigger.header.stamp = this->now();
  trigger.kind = openral_msgs::msg::FailureTrigger::KIND_FORCE;
  trigger.severity = openral_msgs::msg::FailureTrigger::SEVERITY_ABORT;
  trigger.rskill_id = envelope_.rskill_id;
  // Shape matches openral_core.ForceEvidence; a joint torque, hence unit "Nm".
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<double>::max_digits10);
  oss << R"({"kind":"force","joint_or_ee":")" << json_escape(joint) << R"(","measured_n":)"
      << residual_nm << R"(,"limit_n":)" << limit_nm << R"(,"unit":"Nm"})";
  trigger.evidence_json = oss.str();
  failure_pub_->publish(trigger);
  set_safety_status(true, openral_msgs::msg::SafetyStatus::KIND_FORCE,
                    "external joint torque over joint_external_torque_max", envelope_.rskill_id,
                    kNoCorrelationId);
  std_msgs::msg::Empty estop_msg;
  estop_pub_->publish(estop_msg);
  RCLCPP_ERROR(this->get_logger(),
               "safety.force_limit joint=%s external_torque_nm=%g limit_nm=%g rskill_id=%s",
               joint.c_str(), residual_nm, limit_nm, envelope_.rskill_id.c_str());
}

void SafetyKernelLifecycleNode::on_external_estop(const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  if (!fault_latch_) {
    fault_latch_ = true;
    last_estop_at_ = std::chrono::steady_clock::now();
    last_drop_reason_ = "external_estop";
    clear_workspace_commands(ResetReason::kExternalEstop);
    // ADR-0096 — /openral/estop is a bare std_msgs/Empty, so the publisher
    // (deadman watchdog, hardware pendant, dashboard button, the passthrough
    // node) is unknowable from the wire. DROP_EXTERNAL_ESTOP names the topic
    // that latched us rather than claiming a violation kind nobody reported
    // (CLAUDE.md §1.2).
    set_safety_status(true, openral_msgs::msg::SafetyStatus::DROP_EXTERNAL_ESTOP,
                      "external /openral/estop publication latched the kernel", kNoCorrelationId,
                      kNoCorrelationId);
    RCLCPP_WARN(this->get_logger(), "safety.external_estop_received: latching kernel");
  }
}

void SafetyKernelLifecycleNode::on_estop_reset(
    const std_srvs::srv::Trigger::Request::SharedPtr /*request*/,
    const std_srvs::srv::Trigger::Response::SharedPtr response) {
  if (!fault_latch_) {
    response->success = true;
    response->message = "no estop to reset";
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed = now - last_estop_at_;
  const auto cooldown = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(estop_reset_cooldown_s_));
  if (elapsed < cooldown) {
    response->success = false;
    std::ostringstream oss;
    const double sec = std::chrono::duration<double>(elapsed).count();
    oss << "cooldown not elapsed (" << sec << "s < " << estop_reset_cooldown_s_ << "s)";
    response->message = oss.str();
    return;
  }
  fault_latch_ = false;
  last_drop_reason_.clear();
  clear_workspace_commands(ResetReason::kEstopReset);
  // #373: start the external-torque estimate from scratch. A contact that is
  // still there ramps back up within a few observer time constants and trips
  // again; the residual that latched is not carried across the reset.
  reset_force_observer(force_state_);
  force_last_sample_ns_ = 0;
  // ADR-0096 clear transition — the durable value must follow recovery, or a
  // late-joining consumer would read a cleared kernel as still latched.
  set_safety_status(false, openral_msgs::msg::SafetyStatus::DROP_NONE,
                    "estop cleared via /openral/estop_reset", kNoCorrelationId, kNoCorrelationId);
  response->success = true;
  response->message = "estop cleared";
  RCLCPP_INFO(this->get_logger(), "safety.estop_reset succeeded");
}

void SafetyKernelLifecycleNode::publish_diagnostics() {
  diagnostic_msgs::msg::DiagnosticArray arr;
  arr.header.stamp = this->now();
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "openral_safety_kernel";
  status.hardware_id = envelope_.robot_name;
  status.level = fault_latch_ ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
                              : diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = fault_latch_ ? "fault latched" : "passthrough active";
  auto add_kv = [&status](const std::string& k, const std::string& v) {
    diagnostic_msgs::msg::KeyValue kv;
    kv.key = k;
    kv.value = v;
    status.values.push_back(kv);
  };
  add_kv("passed", std::to_string(chunks_passed_));
  add_kv("dropped", std::to_string(chunks_dropped_));
  // A subset of `passed` (#188), not of `dropped`: these chunks were accepted
  // and executed, just slower. The per-event log line is transition-free, so
  // this heartbeat is what makes a robot that has been crawling for a minute
  // visible without grepping for every scaling event.
  add_kv("scaled", std::to_string(chunks_scaled_));
  add_kv("last_drop_reason", last_drop_reason_.empty() ? "-" : last_drop_reason_);
  add_kv("envelope_loaded", envelope_loaded_ ? "true" : "false");
  add_kv("n_dof", std::to_string(envelope_.n_dof));
  add_kv("exemption_provider", provider_status_);
  add_kv("exemption_provider_pairs", std::to_string(provider_pairs_total_));
  // Standing place-declaration state. The refusal logs are transition-gated, so
  // this heartbeat is what makes a persistent refusal (or a live allowance)
  // observable without re-warning at the attachment rate.
  std::string place_region_state{"-"};
  if (place_region_.valid) {
    place_region_state = (place_declaration_live() ? "live:" : "expired:") +
                         place_declaration_target_ +
                         ":geom=" + std::to_string(place_region_.n_geometry);
  } else if (!place_region_refusal_reason_.empty()) {
    place_region_state = place_region_refusal_reason_ + ":" + place_region_refusal_target_;
  }
  add_kv("place_region", place_region_state);
  // Standing grasp-declaration state (ADR-0115 draft), for the same reason.
  std::string grasp_region_state{"-"};
  if (!grasp_allowance_enabled_) {
    grasp_region_state = "off";
  } else if (grasp_region_.valid) {
    grasp_region_state = (grasp_declaration_live() ? "live:" : "expired:") +
                         grasp_declaration_target_ +
                         ":links=" + std::to_string(grasp_region_.link_mask.count()) +
                         (grasp_region_.handover ? ":handover" : "");
  } else if (!grasp_region_refusal_reason_.empty()) {
    grasp_region_state = grasp_region_refusal_reason_ + ":" + grasp_region_refusal_target_;
  }
  add_kv("grasp_region", grasp_region_state);
  if (force_observer_enabled_) {
    add_kv("force_observer", force_estimate_valid_ ? "valid" : "invalid");
    add_kv("force_peak_ratio", std::to_string(force_peak_ratio_));
    force_peak_ratio_ = 0.0;
  }
  arr.status.push_back(status);
  diagnostics_pub_->publish(arr);
  // ADR-0096 / HZ-0096-1 mitigation 2 — refresh the latched status at the
  // same 1 Hz cadence so `header.stamp` is standing evidence the publisher is
  // alive. Without it a consumer applying the staleness rule could not tell a
  // genuinely-latched kernel from a dead publisher's leftover durable sample.
  // The value itself is unchanged; only the stamp moves.
  publish_safety_status_now();
}

void SafetyKernelLifecycleNode::set_safety_status(bool latched, std::uint8_t drop_reason,
                                                  const char* detail, const std::string& rskill_id,
                                                  const std::string& trace_id) {
  if (status_msg_.latched == latched && status_msg_.drop_reason == drop_reason) {
    return;  // not a transition — see the header comment for why this gates
  }
  status_msg_.latched = latched;
  status_msg_.drop_reason = drop_reason;
  status_msg_.detail = detail;
  status_msg_.rskill_id = rskill_id;
  status_msg_.trace_id = trace_id;
  publish_safety_status_now();
}

void SafetyKernelLifecycleNode::publish_safety_status_now() {
  if (status_pub_ == nullptr || !status_pub_->is_activated()) {
    // Not activated: the transition is still recorded in status_msg_ and goes
    // out with the activation publish. Skipping keeps the deactivated-publish
    // warning off the 1 Hz timer.
    return;
  }
  status_msg_.header.stamp = this->now();
  status_pub_->publish(status_msg_);
}

void SafetyKernelLifecycleNode::publish_failure_trigger(const openral_msgs::msg::ActionChunk& chunk,
                                                        const Violation& v) {
  openral_msgs::msg::FailureTrigger trigger;
  trigger.header.stamp = this->now();
  trigger.kind = violation_kind_constant(v.kind);
  trigger.severity = openral_msgs::msg::FailureTrigger::SEVERITY_ABORT;
  trigger.rskill_id = chunk.rskill_id;
  trigger.trace_id = chunk.trace_id;

  // evidence_json — shape matches openral_core.FailureEvidence
  // discriminated-union variants exactly. Hand-built JSON; the
  // receiver (reasoner) round-trips via
  // ``TypeAdapter(FailureEvidence).validate_json(...)``.
  const bool joint_mode =
      chunk.control_mode <= static_cast<std::uint8_t>(ControlMode::kJointTrajectory);
  const auto joint_label = [&]() -> std::string {
    if (v.joint_index == 0xFFFF) {
      return "ee";
    }
    if (joint_mode && collision_joint_names_.size() == envelope_.n_dof &&
        joint_name_to_dof_.size() == envelope_.n_dof &&
        v.joint_index < collision_joint_names_.size() &&
        collision_joint_names_[v.joint_index].find_first_not_of(" \t\r\n\f\v") !=
            std::string::npos) {
      return collision_joint_names_[v.joint_index];
    }
    return "joint_" + std::to_string(v.joint_index);
  };
  std::ostringstream oss;
  switch (v.kind) {
  case ViolationKind::kForce: {
    // ForceEvidence: joint_or_ee, measured_n, limit_n (limit_n must be > 0).
    // For joint-velocity / cartesian-twist speed violations we still
    // route through ForceEvidence — the measured field carries the
    // magnitude that crossed the limit.
    const double measured = std::abs(v.offending_value);
    const double limit =
        std::abs(v.limit_value) > 0.0 ? std::abs(v.limit_value) : 1e-9;  // schema requires > 0
    oss << R"({"kind":"force","joint_or_ee":")" << json_escape(joint_label())
        << R"(","measured_n":)" << measured << R"(,"limit_n":)" << limit << "}";
    break;
  }
  case ViolationKind::kWorkspace: {
    // WorkspaceEvidence: ee_name, measured_xyz, box_min, box_max.
    // For joint-position violations there is no Cartesian xyz; we
    // synthesise a 1-D embedding (offending value on x, limit on
    // box_min.x or box_max.x) so the schema is satisfied. The reasoner
    // sees the field shape; the joint_index and field semantics are
    // carried by ``ee_name`` (e.g. ``"joint_1"``).
    const bool is_cartesian = (v.field[0] == 'w' && v.field[1] == 'o');
    // workspace_xyz field → real Cartesian violation. Other fields are joint.
    // A gripper_range violation carries the gripper's index in joint_index;
    // report the configured end-effector name, which the reasoner reads.
    const bool is_gripper = std::string_view(v.field) == "gripper_range" &&
                            v.joint_index < envelope_.gripper_ee_names.size();
    const std::string ee_name = is_cartesian ? std::string{"end_effector"}
                                : is_gripper ? envelope_.gripper_ee_names[v.joint_index]
                                             : joint_label();
    const double meas = v.offending_value;
    const double limit = v.limit_value;
    // For non-cartesian violations, embed the 1-D bound into the x axis
    // and zero the others. Cartesian violations supply the real xyz
    // semantics through joint_index ∈ {0,1,2} → x/y/z.
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    double box_min_x = 0.0;
    double box_max_x = 0.0;
    if (is_cartesian) {
      const std::size_t axis = static_cast<std::size_t>(v.joint_index % 3);
      if (axis == 0) {
        mx = meas;
      } else if (axis == 1) {
        my = meas;
      } else {
        mz = meas;
      }
      // Use the offending value vs. limit as a 1-D synthetic box on x;
      // downstream consumers care about ee_name + measured_xyz.
      box_min_x = std::min(limit, meas);
      box_max_x = std::max(limit, meas);
    } else {
      mx = meas;
      box_min_x = std::min(limit, meas - 1e-9);
      box_max_x = std::max(limit, meas + 1e-9);
    }
    oss << R"({"kind":"workspace","ee_name":")" << json_escape(ee_name) << R"(","measured_xyz":[)"
        << mx << "," << my << "," << mz << R"(],"box_min":[)" << box_min_x
        << R"(,0.0,0.0],"box_max":[)" << box_max_x << R"(,0.0,0.0]})";
    break;
  }
  case ViolationKind::kController:
  default: {
    // ControllerEvidence: controller_name, state, detail.
    const std::string state =
        (v.sub == ControllerSubKind::kNanInAction)            ? "nan_in_action"
        : (v.sub == ControllerSubKind::kNdofMismatch)         ? "ndof_mismatch"
        : (v.sub == ControllerSubKind::kDimMismatch)          ? "dim_mismatch"
        : (v.sub == ControllerSubKind::kInvalidScale)         ? "invalid_scale"
        : (v.sub == ControllerSubKind::kGripperUnresolved)    ? "gripper_unresolved"
        : (v.sub == ControllerSubKind::kEnvelopeUnconfigured) ? "envelope_unconfigured"
                                                              : "controller_error";
    std::ostringstream detail;
    detail << "field=" << v.field;
    if (v.joint_index != 0xFFFF) {
      detail << " joint=" << joint_label();
    }
    detail << " value=" << v.offending_value << " limit=" << v.limit_value;
    oss << R"({"kind":"controller","controller_name":"openral_safety_kernel","state":")" << state
        << R"(","detail":")" << json_escape(detail.str()) << R"("})";
    break;
  }
  }
  trigger.evidence_json = oss.str();
  failure_pub_->publish(trigger);
}

void SafetyKernelLifecycleNode::load_exemption_provider() {
  voxel_grid_.provider_gate = ProviderGate{};
  provider_.reset();
  provider_links_.clear();
  provider_pairs_total_ = 0;
  const std::string name = this->get_parameter("exemption_provider").as_string();
  if (name.empty()) {
    provider_status_ = "none";
    return;
  }
  // Every refusal leaves the kernel exactly as with no provider (ADR-0121 §4.1), and is
  // announced: logged here, standing on /diagnostics.
  const auto refuse = [&](const std::string& why) {
    provider_.reset();
    provider_links_.clear();
    provider_status_ = "refused: " + why;
    RCLCPP_ERROR(this->get_logger(), "safety.exemption_provider refused name=%s: %s", name.c_str(),
                 why.c_str());
  };
  const std::string hal_mode = this->get_parameter("hal_mode").as_string();
  if (hal_mode != "sim") {
    refuse("sim only (ADR-0121), hal_mode=" + hal_mode);
    return;
  }
  const auto links = this->get_parameter("exemption_provider_links").as_string_array();
  if (links.empty()) {
    refuse("exemption_provider_links is empty");
    return;
  }
  std::vector<std::uint8_t> mask(collision_link_names_.size(), 0);
  for (const auto& link : links) {
    const auto it = std::find(collision_link_names_.begin(), collision_link_names_.end(), link);
    if (it == collision_link_names_.end()) {
      refuse("exemption_provider_links names an unknown link: " + link);
      return;
    }
    mask[static_cast<std::size_t>(it - collision_link_names_.begin())] = 1;
  }
  try {
    if (provider_loader_ == nullptr) {
      provider_loader_ = std::make_unique<pluginlib::ClassLoader<ExemptionProvider>>(
          "openral_safety_kernel", "openral_safety_kernel::ExemptionProvider");
    }
    provider_ = provider_loader_->createSharedInstance(name);
  } catch (const std::exception& e) {
    refuse(std::string("load failed: ") + e.what());
    return;
  }
  if (provider_->api_version() != kExemptionProviderApiVersion) {
    refuse("api_version " + std::to_string(provider_->api_version()) +
           " != " + std::to_string(kExemptionProviderApiVersion));
    return;
  }
  bool ok = false;
  try {
    ok = provider_->initialize(collision_model_, this->get_node_parameters_interface(),
                               this->get_logger().get_child("exemption_provider"));
  } catch (const std::exception& e) {
    refuse(std::string("initialize threw: ") + e.what());
    return;
  }
  if (!ok) {
    refuse("initialize declined");
    return;
  }
  provider_links_ = std::move(mask);
  provider_link_bits_.reset();
  for (std::size_t l = 0; l < provider_links_.size() && l < kMaxGraspMaskLinks; ++l) {
    provider_link_bits_[l] = provider_links_[l] != 0;
  }
  // The hands for the provider's DeclarationView, from the open ADR-0115 allowlist.
  provider_hands_ = TrackedGraspTable{};
  provider_hands_valid_ =
      grasp_allowance_enabled_ &&
      build_grasp_hands(collision_model_, grasp_allowlist_, provider_hands_, {});
  provider_meas_scratch_.link_world.assign(collision_model_.n_links, Transform{});
  provider_q_fk_.assign(q_fk_.size(), 0.0);
  provider_status_ = std::string("loaded: ") + provider_->name();
  RCLCPP_WARN(this->get_logger(),
              "safety.exemption_provider loaded name=%s provider=%s links=%zu (ADR-0121, sim only)",
              name.c_str(), provider_->name(), links.size());
}

bool SafetyKernelLifecycleNode::load_collision_model(std::string& error) {
  // Reset to the disabled state so a re-configure can never leak a stale model.
  collision_model_ = CollisionModel{};
  collision_link_names_.clear();
  collision_scratch_.link_world.clear();
  binding_scratch_.link_world.clear();
  collision_required_dof_ = 0;
  self_collision_enabled_ = this->get_parameter("self_collision_enabled").as_bool();

  // Voxel (dense occupancy grid) config. Pre-size the occupancy buffer once so
  // the subscription callback never reallocates and the view pointer is stable.
  world_voxel_enabled_ = this->get_parameter("world_voxel_enabled").as_bool();
  world_voxel_margin_m_ = this->get_parameter("world_voxel_margin_m").as_double();
  const double deadline_ms = this->get_parameter("world_voxel_deadline_ms").as_double();
  const double data_age_budget_ms =
      this->get_parameter("world_voxel_data_age_budget_ms").as_double();
  world_voxel_deadline_s_ = deadline_ms / 1000.0;
  world_voxel_data_age_budget_s_ = data_age_budget_ms / 1000.0;
  // Hard caps, mirroring DeployRuntime (hazard log Entries 033/034). Refused,
  // never clamped: a cap exceeded here means a launch path bypassed the
  // schema, and the operator must see it rather than run on a guess.
  if (world_voxel_enabled_) {
    if (!(deadline_ms > 0.0 && deadline_ms <= kMaxWorldVoxelDeadlineMs)) {
      error = "world_voxel_deadline_ms=" + std::to_string(deadline_ms) + " is outside (0, " +
              std::to_string(kMaxWorldVoxelDeadlineMs) +
              "]: the kernel would check chunks against a voxel grid that old";
      return false;
    }
    if (!(data_age_budget_ms > 0.0 && data_age_budget_ms <= kMaxWorldVoxelDataAgeBudgetMs)) {
      error = "world_voxel_data_age_budget_ms=" + std::to_string(data_age_budget_ms) +
              " is outside (0, " + std::to_string(kMaxWorldVoxelDataAgeBudgetMs) +
              "]: the kernel would certify chunks against a world that old (0 no longer "
              "disables the check)";
      return false;
    }
  }
  voxel_source_known_ = false;
  world_voxel_max_cells_ =
      static_cast<std::size_t>(this->get_parameter("world_voxel_max_cells").as_int());
  voxel_received_ = false;
  voxel_overflow_ = false;
  voxel_grid_ = VoxelGrid{};
  voxel_occupancy_.assign(world_voxel_max_cells_, 0);
  voxel_grid_.occupancy = voxel_occupancy_.data();

  // Attached-payload config (ADR-0092). Pre-size the fixed-capacity attached
  // model + label buffer once so the world-state callback fills in place and the
  // hot path never allocates.
  attached_collision_enabled_ = this->get_parameter("attached_collision_enabled").as_bool();
  attached_collision_margin_m_ = this->get_parameter("attached_collision_margin_m").as_double();
  {
    const std::int64_t advisory_cap =
        this->get_parameter("place_advisory_max_consecutive").as_int();
    // A negative cap is a misconfiguration, and the safe reading of one is "no
    // advisory band" rather than "an unbounded one".
    place_advisory_max_consecutive_ =
        advisory_cap > 0 ? static_cast<std::uint64_t>(advisory_cap) : 0U;
  }
  advisory_refusals_ = 0;
  {
    // A negative band is a misconfiguration, and the safe reading is "no
    // band".
    const double proximity = this->get_parameter("collision_scale_proximity_m").as_double();
    collision_scale_proximity_m_ = proximity > 0.0 ? proximity : 0.0;
    const double k = this->get_parameter("collision_scale_k").as_double();
    collision_scale_k_ = k > 0.0 ? k : 0.0;
    // The floor must be strictly positive and at most 1. Zero is not a "very
    // slow" setting — it is a full stop that emits no E-stop, no
    // FailureTrigger and no latched status, while /openral/safety_status still
    // reports "chunk accepted". Above 1 would SPEED THE ROBOT UP. Either is a
    // misconfiguration of a safety surface, so neither is quietly clamped into
    // range: the band is refused outright and the kernel says why, leaving
    // today's behaviour rather than an invented one.
    collision_scale_min_ = this->get_parameter("collision_scale_min").as_double();
    if (collision_scale_proximity_m_ > 0.0 &&
        !(collision_scale_min_ > 0.0 && collision_scale_min_ <= 1.0)) {
      RCLCPP_ERROR(this->get_logger(),
                   "collision_scale_min=%g is outside (0, 1] — graded velocity scaling is "
                   "DISABLED. A floor of 0 is a stop that reports itself as an accepted "
                   "chunk; above 1 would speed the robot up.",
                   collision_scale_min_);
      collision_scale_proximity_m_ = 0.0;
      collision_scale_min_ = 1.0;
    }
    if (collision_scale_proximity_m_ > 0.0) {
      RCLCPP_INFO(this->get_logger(),
                  "safety.collision_scaling armed: band=%g m k=%g floor=%g (scale at margin=%g)",
                  collision_scale_proximity_m_, collision_scale_k_, collision_scale_min_,
                  velocity_scale_for(0.0));
    }
  }
  attached_collision_deadline_s_ =
      this->get_parameter("attached_collision_deadline_ms").as_double() / 1000.0;
  attached_max_objects_ =
      static_cast<std::size_t>(this->get_parameter("attached_max_objects").as_int());
  attached_max_primitives_ =
      static_cast<std::size_t>(this->get_parameter("attached_max_primitives").as_int());
  attached_max_touch_links_ =
      static_cast<std::size_t>(this->get_parameter("attached_max_touch_links").as_int());
  attached_received_ = false;
  attached_overflow_ = false;
  attached_revision_ = 0;
  attached_contact_snapshot_pending_ = false;
  attached_contact_active_ = false;
  support_witness_live_ = 0;
  support_witness_keys_.assign(std::min<std::size_t>(attached_max_objects_, 8),
                               SupportWitnessKey{});
  support_witness_max_patch_radius_m_ =
      this->get_parameter("support_witness_max_patch_radius_m").as_double();
  support_witness_max_penetration_m_ =
      this->get_parameter("support_witness_max_penetration_m").as_double();
  attached_contact_mask_.assign(world_voxel_max_cells_, 0);
  attached_contact_distance_.assign(attached_max_objects_ * world_voxel_max_cells_,
                                    std::numeric_limits<double>::infinity());
  voxel_grid_.attached_contact_mask = attached_contact_mask_.data();
  voxel_grid_.attached_contact_distance = attached_contact_distance_.data();
  voxel_grid_.attached_contact_stride = world_voxel_max_cells_;
  voxel_grid_.attached_contact_tolerance =
      this->get_parameter("attached_contact_tolerance_m").as_double();
  voxel_grid_.support_witness_live = 0;
  voxel_grid_.place_region = PlaceApproachRegion{};
  place_region_ = PlaceApproachRegion{};
  voxel_grid_.grasp_region = GraspTargetRegion{};
  grasp_region_ = GraspTargetRegion{};
  grasp_declaration_stamp_ns_ = 0;
  grasp_declaration_timeout_s_ = 0.0;
  grasp_declaration_target_.clear();
  grasp_retired_.clear();
  grasp_retired_overflow_logged_ = false;
  tracked_retired_.clear();
  tracked_n_ = 0;
  cover_pass_n_ = 0;
  tracked_have_revision_ = false;
  tracked_revision_ = 0;
  tracked_refusal_reason_.clear();
  tracked_set_frame_.clear();
  tracked_grasp_enabled_ = false;
  tracked_grasp_ = TrackedGraspTable{};
  voxel_grid_.tracked_grasp = nullptr;
  tracked_cover_dirty_ = true;
  tracked_decl_active_ = false;
  tracked_decl_key_.clear();
  tracked_grasp_logged_n_ = static_cast<std::size_t>(-1);
  workspace_explained_enabled_ = false;
  workspace_explained_ = WorkspaceExplanation{};
  support_cells_marked_ = 0;
  support_cells_dirty_ = true;
  voxel_grid_.workspace_explained = nullptr;
  static_cells_marked_ = 0;
  static_cells_for_ = VoxelGrid{};
  static_cells_for_valid_ = false;
  static_cells_lattice_ok_ = false;
  workspace_explained_logged_n_ = static_cast<std::size_t>(-1);
  bound_.fill(BoundPayload{});
  grasp_latched_tracked_ = false;
  grasp_latched_epoch_ = 0;
  grasp_wrong_claim_ = false;
  grasp_region_refusal_reason_.clear();
  grasp_region_refusal_target_.clear();
  grasp_latched_region_ = GraspTargetRegion{};
  grasp_latched_target_.clear();
  grasp_latched_stamp_ns_ = 0;
  grasp_latched_ = false;
  grasp_latched_moved_warned_ = false;
  workspace_region_ = WorkspaceContactRegion{};
  voxel_grid_.workspace_region = WorkspaceContactRegion{};
  workspace_target_.clear();
  workspace_refusal_reason_.clear();
  workspace_pairs_total_ = 0;
  place_declaration_stamp_ns_ = 0;
  place_declaration_timeout_s_ = 0.0;
  place_declaration_target_.clear();
  voxel_frame_id_.clear();
  attached_model_ = AttachedModel{};
  attached_model_.objects.assign(attached_max_objects_, AttachedObject{});
  attached_model_.primitives.assign(attached_max_primitives_, AttachedPrimitive{});
  attached_model_.touch_links.assign(attached_max_touch_links_, 0);
  // Stage-2 store for carried payloads (#266). Pre-sized to the worst case —
  // every primitive refined, each with a full-budget hull — so the ingest path
  // fills it in place and the hot path never allocates, the same contract the
  // fixed link hulls are under. 16 primitives x 320 vertices x 24 B is ~123 kB.
  attached_model_.hulls.assign(attached_max_primitives_, LinkHull{});
  attached_model_.hull_vertices.assign(
      attached_max_primitives_ * static_cast<std::size_t>(kMaxTightHullVertices), Vec3{});
  attached_labels_.assign(attached_max_objects_, std::string{});
  attached_evidence_kinds_.assign(attached_max_objects_, std::string{});
  attached_evidence_refs_.assign(attached_max_objects_, std::string{});
  attached_binding_.assign(attached_max_objects_, AttachedBinding{});
  bound_slot_.assign(attached_max_objects_, -1);
  attached_ingest_scratch_.clear();
  attached_ingest_scratch_.reserve(attached_max_objects_);
  // Sized once, here, and never resized again: `place_region_.geometry` is a
  // view into this buffer and the hot path reads it (ADR-0098).
  place_geometry_.assign(kMaxPlaceTargetPrimitives, AttachedPrimitive{});
  place_geometry_scratch_.clear();
  place_geometry_scratch_.reserve(kMaxPlaceTargetPrimitives);

  // The robot collision model is needed for any geometric check; skip loading
  // only when all of them are disabled.
  if (!self_collision_enabled_ && !world_voxel_enabled_ && !attached_collision_enabled_) {
    return true;
  }

  self_collision_margin_m_ = this->get_parameter("self_collision_margin_m").as_double();
  const auto n_links = static_cast<std::size_t>(this->get_parameter("collision_n_links").as_int());
  if (n_links == 0) {
    error = "self_collision_enabled but collision_n_links == 0";
    return false;
  }

  const auto parent = this->get_parameter("collision_parent").as_integer_array();
  const auto kind = this->get_parameter("collision_joint_kind").as_integer_array();
  const auto dof = this->get_parameter("collision_dof_index").as_integer_array();
  const auto origin = this->get_parameter("collision_origin_xyzrpy").as_double_array();
  const auto axis = this->get_parameter("collision_axis").as_double_array();
  const auto cap_link = this->get_parameter("collision_capsule_link").as_integer_array();
  const auto cap_r = this->get_parameter("collision_capsule_radius").as_double_array();
  const auto cap_h = this->get_parameter("collision_capsule_half_length").as_double_array();
  const auto cap_o = this->get_parameter("collision_capsule_origin_xyzrpy").as_double_array();
  const auto box_link = this->get_parameter("collision_box_link").as_integer_array();
  const auto box_he = this->get_parameter("collision_box_half_extents").as_double_array();
  const auto box_o = this->get_parameter("collision_box_origin_xyzrpy").as_double_array();
  const auto box_hull = this->get_parameter("collision_box_hull").as_integer_array();
  const auto hull_lo = this->get_parameter("collision_hull_dop_lo").as_double_array();
  const auto hull_hi = this->get_parameter("collision_hull_dop_hi").as_double_array();
  const auto hull_first = this->get_parameter("collision_hull_vertex_first").as_integer_array();
  const auto hull_count = this->get_parameter("collision_hull_vertex_count").as_integer_array();
  const auto hull_verts = this->get_parameter("collision_hull_vertices").as_double_array();
  const auto pairs = this->get_parameter("collision_allowed_pairs").as_integer_array();
  const auto names = this->get_parameter("collision_link_names").as_string_array();
  const auto mimic_mult = this->get_parameter("collision_mimic_multiplier").as_double_array();
  const auto mimic_off = this->get_parameter("collision_mimic_offset").as_double_array();

  // Per-link arrays are sized to n_links; capsule/box arrays are sized to the
  // (independent) primitive count — a link may carry zero, one, or several.
  const std::size_t n_caps = cap_r.size();
  const std::size_t n_boxes = box_link.size();
  if (parent.size() != n_links || kind.size() != n_links || dof.size() != n_links ||
      origin.size() != 6 * n_links || axis.size() != 3 * n_links || names.size() != n_links ||
      cap_link.size() != n_caps || cap_h.size() != n_caps || cap_o.size() != 6 * n_caps ||
      box_he.size() != 3 * n_boxes || box_o.size() != 6 * n_boxes || pairs.size() % 2 != 0) {
    error = "collision_* array shapes disagree with collision_n_links / primitive count";
    return false;
  }
  // Tight geometry is all-or-nothing per model: either no box declares any, or
  // `collision_box_hull` names one entry per box. A partially-shaped set is a
  // producer bug, and refusing it keeps the shipped narrow phase everywhere
  // rather than half-applying a tightening nobody checked.
  const std::size_t n_hulls = hull_count.size();
  if (!box_hull.empty()) {
    if (box_hull.size() != n_boxes || hull_first.size() != n_hulls ||
        hull_lo.size() != static_cast<std::size_t>(kDopAxes) * n_hulls ||
        hull_hi.size() != static_cast<std::size_t>(kDopAxes) * n_hulls ||
        hull_verts.size() % 3 != 0) {
      error = "collision_hull_* array shapes disagree with the box / hull count";
      return false;
    }
  } else if (n_hulls != 0 || !hull_lo.empty() || !hull_hi.empty() || !hull_verts.empty()) {
    error = "collision_hull_* supplied without collision_box_hull";
    return false;
  }
  // Mimic relation: all-or-nothing per model, one entry per link.
  if (mimic_mult.size() != mimic_off.size() ||
      (!mimic_mult.empty() && mimic_mult.size() != n_links)) {
    error = "collision_mimic_* array shapes disagree with collision_n_links";
    return false;
  }

  CollisionModel m;
  m.n_links = n_links;
  m.parent.resize(n_links);
  m.joint_kind.resize(n_links);
  m.dof_index.resize(n_links);
  m.origin.resize(n_links);
  m.axis.resize(n_links);
  m.mimic_multiplier.assign(mimic_mult.begin(), mimic_mult.end());
  m.mimic_offset.assign(mimic_off.begin(), mimic_off.end());
  for (std::size_t i = 0; i < n_links; ++i) {
    m.parent[i] = static_cast<int>(parent[i]);
    if (kind[i] < 0 || kind[i] > static_cast<std::int64_t>(JointKind::kMimicPrismatic)) {
      error = "collision_joint_kind carries an unknown joint code";
      return false;
    }
    m.joint_kind[i] = static_cast<JointKind>(static_cast<std::uint8_t>(kind[i]));
    m.dof_index[i] = static_cast<int>(dof[i]);
    if (is_mimic(m.joint_kind[i])) {
      // A mimic link reads another joint's column: it needs that column, a
      // finite relation and a non-zero multiplier (zero is a fixed link
      // wearing a moving kind), all of which the manifest schema also
      // refuses — the kernel does not trust the producer for it.
      if (m.dof_index[i] < 0 || m.mimic_multiplier.empty() ||
          !std::isfinite(m.mimic_multiplier[i]) || !std::isfinite(m.mimic_offset[i]) ||
          m.mimic_multiplier[i] == 0.0) {
        error = "a mimic link (collision_joint_kind 3/4) needs a dof column and a finite, "
                "non-zero collision_mimic_multiplier / finite collision_mimic_offset";
        return false;
      }
    }
    m.origin[i] = transform_from_xyz_rpy(origin[6 * i + 0], origin[6 * i + 1], origin[6 * i + 2],
                                         origin[6 * i + 3], origin[6 * i + 4], origin[6 * i + 5]);
    m.axis[i] = Vec3{axis[3 * i + 0], axis[3 * i + 1], axis[3 * i + 2]};
    if (m.dof_index[i] >= 0) {
      const std::size_t needed = static_cast<std::size_t>(m.dof_index[i]) + 1;
      if (needed > collision_required_dof_) {
        collision_required_dof_ = needed;
      }
    }
  }
  m.capsule_link.resize(n_caps);
  m.capsules.resize(n_caps);
  for (std::size_t c = 0; c < n_caps; ++c) {
    const int link = static_cast<int>(cap_link[c]);
    if (link < 0 || static_cast<std::size_t>(link) >= n_links) {
      error = "collision_capsule_link out of range";
      return false;
    }
    m.capsule_link[c] = link;
    m.capsules[c].radius = cap_r[c];
    m.capsules[c].half_length = cap_h[c];
    m.capsules[c].origin =
        transform_from_xyz_rpy(cap_o[6 * c + 0], cap_o[6 * c + 1], cap_o[6 * c + 2],
                               cap_o[6 * c + 3], cap_o[6 * c + 4], cap_o[6 * c + 5]);
  }
  m.box_link.resize(n_boxes);
  m.boxes.resize(n_boxes);
  for (std::size_t b = 0; b < n_boxes; ++b) {
    const int link = static_cast<int>(box_link[b]);
    if (link < 0 || static_cast<std::size_t>(link) >= n_links) {
      error = "collision_box_link out of range";
      return false;
    }
    m.box_link[b] = link;
    m.boxes[b].half_extents = Vec3{box_he[3 * b + 0], box_he[3 * b + 1], box_he[3 * b + 2]};
    m.boxes[b].origin =
        transform_from_xyz_rpy(box_o[6 * b + 0], box_o[6 * b + 1], box_o[6 * b + 2],
                               box_o[6 * b + 3], box_o[6 * b + 4], box_o[6 * b + 5]);
  }
  if (!box_hull.empty()) {
    m.hulls.resize(n_hulls);
    for (std::size_t h = 0; h < n_hulls; ++h) {
      m.hulls[h].vertex_first = static_cast<int>(hull_first[h]);
      m.hulls[h].vertex_count = static_cast<int>(hull_count[h]);
      for (int i = 0; i < kDopAxes; ++i) {
        m.hulls[h].dop_lo[i] = hull_lo[h * kDopAxes + static_cast<std::size_t>(i)];
        m.hulls[h].dop_hi[i] = hull_hi[h * kDopAxes + static_cast<std::size_t>(i)];
      }
    }
    m.hull_vertices.resize(hull_verts.size() / 3);
    for (std::size_t v = 0; v < m.hull_vertices.size(); ++v) {
      m.hull_vertices[v] = Vec3{hull_verts[3 * v], hull_verts[3 * v + 1], hull_verts[3 * v + 2]};
    }
    m.box_hull.resize(n_boxes);
    for (std::size_t b = 0; b < n_boxes; ++b) {
      const std::int64_t h = box_hull[b];
      if (h < -1 || h >= static_cast<std::int64_t>(n_hulls)) {
        error = "collision_box_hull out of range";
        return false;
      }
      m.box_hull[b] = static_cast<int>(h);
    }
    // The containment proof, enforced at load and never assumed: every declared
    // representation must sit inside the OBB whose broad-phase window it will be
    // checked in. Fail-closed -- a model that cannot prove it does not configure.
    std::size_t offending = 0;
    const TightGeometryStatus status = validate_tight_geometry(m, offending);
    if (status != TightGeometryStatus::kOk) {
      error = std::string("tight collision geometry rejected for box ") +
              std::to_string(offending) + ": " + tight_geometry_status_reason(status);
      return false;
    }
  }

  for (std::size_t k = 0; k + 1 < pairs.size(); k += 2) {
    m.allowed_pairs.emplace_back(static_cast<int>(pairs[k]), static_cast<int>(pairs[k + 1]));
  }

  collision_model_ = std::move(m);
  collision_link_names_.assign(names.begin(), names.end());
  collision_scratch_.link_world.resize(n_links);
  binding_scratch_.link_world.resize(n_links);
  return true;
}

double SafetyKernelLifecycleNode::velocity_scale_for(double slack_m) const noexcept {
  if (collision_scale_proximity_m_ <= 0.0 || !std::isfinite(slack_m) ||
      slack_m >= collision_scale_proximity_m_) {
    return 1.0;
  }
  // `slack_m` is clearance ABOVE the pair's own gate margin, so it is >= 0 for
  // anything that did not trip. exp() is 1.0 at the top of the band and decays
  // toward the margin; the floor bounds it away from an unlogged stop.
  const double raw = std::exp(collision_scale_k_ * (slack_m - collision_scale_proximity_m_));
  return std::clamp(raw, collision_scale_min_, 1.0);
}

void SafetyKernelLifecycleNode::publish_collision_failure(
    const openral_msgs::msg::ActionChunk& chunk, const char* collision_kind,
    const std::string& link_a, const std::string& link_b, int horizon_step, const CollisionHit& hit,
    const std::vector<double>& joint_positions) {
  openral_msgs::msg::FailureTrigger trigger;
  trigger.header.stamp = this->now();
  trigger.kind = openral_msgs::msg::FailureTrigger::KIND_COLLISION;
  trigger.severity = openral_msgs::msg::FailureTrigger::SEVERITY_ABORT;
  trigger.rskill_id = chunk.rskill_id;
  trigger.trace_id = chunk.trace_id;

  // Shape matches openral_core.CollisionEvidence (kind="collision").
  // `max_digits10` is load-bearing, not tidiness: the default 6 significant
  // digits round a joint angle to ~1e-6 rad, which at a metre of reach is
  // millimetres of end-effector error — the same order as the distances this
  // evidence exists to adjudicate. Every number here must round-trip exactly.
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<double>::max_digits10);
  oss << R"({"kind":"collision","collision_kind":")" << json_escape(collision_kind)
      << R"(","link_a":")" << json_escape(link_a) << R"(","link_b_or_object":")"
      << json_escape(link_b) << R"(","horizon_step":)" << horizon_step << R"(,"min_distance_m":)"
      << hit.min_distance << R"(,"joint_positions_rad":[)";
  for (std::size_t i = 0; i < joint_positions.size(); ++i) {
    if (i != 0) {
      oss << ',';
    }
    oss << joint_positions[i];
  }
  oss << ']';
  constexpr std::size_t kEvidenceStringMax = 256;
  const bool joint_names_usable =
      collision_joint_names_.size() == joint_positions.size() &&
      std::all_of(
          collision_joint_names_.begin(), collision_joint_names_.end(),
          [](const auto& name) { return !name.empty() && name.size() <= kEvidenceStringMax; });
  if (joint_names_usable) {
    oss << R"(,"joint_names":[)";
    for (std::size_t i = 0; i < collision_joint_names_.size(); ++i) {
      if (i != 0) {
        oss << ',';
      }
      oss << '"' << json_escape(collision_joint_names_[i]) << '"';
    }
    oss << ']';
  }
  if (std::isfinite(hit.sweep_min_distance)) {
    oss << R"(,"sweep_min_distance_m":)" << hit.sweep_min_distance;
  }
  if (const char* reason = workspace_denial_reason(hit.workspace_denied); reason != nullptr) {
    oss << R"(,"workspace_denial_reason":")" << reason << '"';
  }
  oss << R"(,"workspace_contact_active":)"
      << (voxel_grid_.workspace_region.valid ? "true" : "false");
  const bool grasp_active =
      tracked_grasp_enabled_ ? tracked_grasp_.n_objects > 0 : voxel_grid_.grasp_region.valid;
  oss << R"(,"grasp_exemption_active":)" << (grasp_active ? "true" : "false")
      << R"(,"depth_is_box_bound":)" << (hit.depth_is_box_bound ? "true" : "false");
  // Expose accepted wire provenance only when the named pair contains that
  // attached object. Keep both values paired and omit malformed or overlong
  // producer text rather than truncating it into a different reference.
  for (std::size_t i = 0; i < attached_labels_.size(); ++i) {
    if (attached_labels_[i].empty()) {
      continue;
    }
    const std::string attached_name = "attached:" + attached_labels_[i];
    if (link_a != attached_name && link_b != attached_name) {
      continue;
    }
    const auto& evidence_kind = attached_evidence_kinds_[i];
    const auto& evidence_ref = attached_evidence_refs_[i];
    if (!evidence_kind.empty() && !evidence_ref.empty() &&
        evidence_kind.size() <= kEvidenceStringMax && evidence_ref.size() <= kEvidenceStringMax) {
      oss << R"(,"payload_evidence_kind":")" << json_escape(evidence_kind)
          << R"(","payload_evidence_ref":")" << json_escape(evidence_ref) << '"';
    }
    break;
  }
  // WHICH grid this index addresses (#275). The cell is named only as
  // `voxel_<n>`, an index into a grid the kernel does not republish, and the
  // published window is snapped to the source lattice -- so a base drift
  // across one cell boundary shifts the whole window and the same index names
  // a different cell downstream. A consumer matching on stamps cannot separate
  // two grids inside one `/clock` tick, which quantises every stamp in the
  // graph; the origin is exact. Disclosure only: nothing here gates, and the
  // value is the grid the check above actually ran against.
  if (link_b.rfind("voxel_", 0) == 0) {
    oss << R"(,"world_grid_origin_m":[)" << voxel_grid_.pose.t.x << ',' << voxel_grid_.pose.t.y
        << ',' << voxel_grid_.pose.t.z << ']';
  }
  oss << '}';
  trigger.evidence_json = oss.str();
  failure_pub_->publish(trigger);
}

void SafetyKernelLifecycleNode::on_world_voxels(
    const openral_msgs::msg::OccupancyVoxels::SharedPtr msg) {
  if (msg == nullptr) {
    return;
  }
  tracked_cover_dirty_ = true;  // #418 K2: the cover check runs once per grid
  support_cells_dirty_ = true;  // #418 K7: so does the connected support layer
  const std::size_t cells = static_cast<std::size_t>(msg->size_x) * msg->size_y * msg->size_z;
  // The grid's lattice is the source map's, so `orientation` is load-bearing
  // geometry, not metadata. An unset field is the all-zero quaternion, which is
  // not a rotation: reading it as identity would place every obstacle somewhere
  // the robot is not, and a world check against a misplaced map is fail-OPEN.
  // Refuse it exactly as an over-large or malformed grid is refused.
  if (msg->occupancy.size() != cells || cells > world_voxel_max_cells_ || msg->resolution <= 0.0 ||
      !is_unit_quaternion(msg->orientation)) {
    voxel_overflow_ = true;
    voxel_received_ = true;
    voxel_stamp_ = this->now();
    return;
  }
  voxel_overflow_ = false;
  std::copy(msg->occupancy.begin(), msg->occupancy.end(), voxel_occupancy_.begin());
  voxel_grid_.occupancy = voxel_occupancy_.data();
  voxel_grid_.pose = transform_from_translation_quat(msg->origin.x, msg->origin.y, msg->origin.z,
                                                     msg->orientation.x, msg->orientation.y,
                                                     msg->orientation.z, msg->orientation.w);
  voxel_grid_.resolution = msg->resolution;
  voxel_grid_.sx = static_cast<int>(msg->size_x);
  voxel_grid_.sy = static_cast<int>(msg->size_y);
  voxel_grid_.sz = static_cast<int>(msg->size_z);
  if (workspace_explained_enabled_ || payload_own_enabled_) {
    refresh_static_cells();  // #418 K7 / OT-10: once per change of the grid's lattice
  }
  // A grid that re-frames invalidates any region declared against the old frame
  // — the allowance would otherwise apply to a volume in the wrong place.
  if (msg->header.frame_id != voxel_frame_id_) {
    if (place_region_.valid) {
      RCLCPP_WARN(this->get_logger(),
                  "safety.place_region_dropped reason=grid_frame_changed grid_frame=%s",
                  msg->header.frame_id.c_str());
    }
    voxel_frame_id_ = msg->header.frame_id;
    place_region_ = PlaceApproachRegion{};
    if (grasp_identity_armed()) {
      retire_grasp_declaration("grid_frame_changed");
    }
  }
  voxel_received_ = true;
  voxel_stamp_ = this->now();
  // Zero = the producer said nothing about when its world was captured.
  voxel_source_known_ = msg->source_stamp.sec != 0 || msg->source_stamp.nanosec != 0;
  if (voxel_source_known_) {
    voxel_source_stamp_ = rclcpp::Time(msg->source_stamp, this->get_clock()->get_clock_type());
  }
}

void SafetyKernelLifecycleNode::on_world_state(
    const openral_msgs::msg::WorldStateStamped::SharedPtr msg) {
  if (msg == nullptr) {
    return;
  }
  // Issue #418 K1: the tracker's set first, so the binding check below reads
  // this snapshot's set, whatever happens to the attachments.
  ingest_tracked_objects(*msg);
  // Parse the wire attachments into the by-name ingest scratch, then validate +
  // resolve into the fixed-capacity attached model. Any malformed shape,
  // capacity overflow, or unknown attach/touch link fails closed: the model is
  // emptied and the world state is marked invalid so the next candidate action
  // is dropped until a clean message lands (never a silently-unchecked payload).
  const auto clock_type = this->get_clock()->get_clock_type();
  const bool stamp_valid = msg->attachment_stamp_ns > 0;
  const rclcpp::Time producer_stamp(stamp_valid ? msg->attachment_stamp_ns : 0, clock_type);
  const auto fail_closed = [this, producer_stamp](const char* reason) {
    // Never silent (CLAUDE.md §1.4): a producer stuck on a message the kernel
    // refuses looks, from outside, like a kernel that lost its payload model.
    const auto wall = std::chrono::steady_clock::now();
    if (wall - attached_fail_closed_last_warn_ >= std::chrono::seconds(1)) {
      attached_fail_closed_last_warn_ = wall;
      RCLCPP_WARN(this->get_logger(), "safety.attachment_fail_closed reason=%s objects=%zu", reason,
                  attached_model_.n_objects);
    }
    attached_model_.n_objects = 0;
    attached_overflow_ = true;
    attached_received_ = true;
    attached_stamp_ = producer_stamp;
    attached_contact_snapshot_pending_ = false;
    attached_contact_active_ = false;
    support_witness_live_ = 0;
    std::fill(support_witness_keys_.begin(), support_witness_keys_.end(), SupportWitnessKey{});
    std::fill(attached_contact_mask_.begin(), attached_contact_mask_.end(), 0);
    // A message we do not trust for the payload model is not one to trust for
    // the region scoped to that payload either.
    place_region_ = PlaceApproachRegion{};
    if (grasp_identity_armed()) {
      retire_grasp_declaration("attachment_rejected");
    }
  };

  const auto& wire = msg->attached_objects;
  if (!stamp_valid) {
    fail_closed("no_attachment_stamp");
    return;
  }
  if (wire.size() > attached_max_objects_) {
    fail_closed("too_many_objects");
    return;
  }
  // Total-primitive cap across every attached object (early fail-closed before
  // the by-name ingest; ingest_attached_objects re-checks incrementally).
  std::size_t total_primitives = 0;
  for (const auto& obj : wire) {
    total_primitives += obj.primitives.size();
  }
  if (total_primitives > attached_max_primitives_) {
    fail_closed("too_many_primitives");
    return;
  }

  attached_ingest_scratch_.clear();
  for (const auto& obj : wire) {
    AttachedObjectInput in;
    // Object pose in the attach-link frame; each primitive composes on top.
    in.pose_in_link = transform_from_translation_quat(
        obj.pose_in_link.position.x, obj.pose_in_link.position.y, obj.pose_in_link.position.z,
        obj.pose_in_link.orientation.x, obj.pose_in_link.orientation.y,
        obj.pose_in_link.orientation.z, obj.pose_in_link.orientation.w);
    in.attach_link = obj.attach_link;
    in.touch_links.assign(obj.touch_links.begin(), obj.touch_links.end());
    // Support-contact attestation (ADR-0092 D6). World State may attest that
    // this payload rests on a named surface; the kernel bounds what it will
    // accept. An attestation outside those bounds is not silently downgraded to
    // "no witness" — it fails the whole message closed, because a producer that
    // over-claims here is not one to trust with the rest of the payload model.
    in.has_support_witness = obj.support_contact_valid;
    if (in.has_support_witness) {
      const auto& witness = obj.support_contact;
      if (witness.patch_radius_m > support_witness_max_patch_radius_m_ ||
          witness.max_penetration_m > support_witness_max_penetration_m_) {
        RCLCPP_WARN(this->get_logger(),
                    "safety.support_witness_rejected object=%s patch_m=%g penetration_m=%g",
                    obj.object_id.c_str(), witness.patch_radius_m, witness.max_penetration_m);
        fail_closed("support_witness_out_of_bounds");
        return;
      }
      in.support_point = Vec3{witness.contact_point_in_object.x, witness.contact_point_in_object.y,
                              witness.contact_point_in_object.z};
      in.support_normal =
          Vec3{witness.contact_normal_in_object.x, witness.contact_normal_in_object.y,
               witness.contact_normal_in_object.z};
      in.support_patch_radius = witness.patch_radius_m;
      in.support_max_penetration = witness.max_penetration_m;
    }
    // Contact-force attestation (ADR-0100). Unlike the support witness above,
    // an out-of-bounds value here does NOT fail the message closed: this
    // witness licenses nothing, so a producer over-claiming on it cannot buy
    // anything, and refusing the whole payload model would make an additive
    // conservatism feature able to stop the robot by being malformed. It is
    // dropped instead, which lands on the shipped geometry-only behaviour.
    in.has_contact_force_witness = obj.contact_force_valid;
    if (in.has_contact_force_witness) {
      const auto& force = obj.contact_force;
      // The magnitude is Newtons ONLY under a named calibration (survey §21.7).
      // Both conditions are resolved here so the hot path reads two bools.
      in.contact_force_calibrated =
          force.magnitude_calibrated && has_calibration_name(force.calibration_ref);
      // Scoping: the measured contact must be against the surface the
      // declaration named. `target_id` matches the declared target outright, or
      // names a surface that is part of it — the same rule the support
      // witness's `support_id` is under.
      // Read off THIS message, never off `place_declaration_target_`: the
      // declaration is ingested after the payloads (it is resolved against the
      // objects just accepted), so the member still holds the previous
      // snapshot's target here. An empty or inactive declaration matches
      // nothing, which lands on the shipped geometry-only behaviour.
      const std::string declared_target =
          (msg->place_declaration_valid && msg->place_declaration.active)
              ? msg->place_declaration.target_id
              : std::string{};
      in.contact_force_target_matches =
          !declared_target.empty() && (force.target_id == declared_target ||
                                       force.target_id.rfind(declared_target + ":", 0) == 0);
      in.contact_force_magnitude = force.magnitude_n;
      in.contact_force_stamp_ns = force.stamp_ns;
      if (!std::isfinite(in.contact_force_magnitude) || in.contact_force_magnitude < 0.0) {
        in.has_contact_force_witness = false;
        in.contact_force_calibrated = false;
        in.contact_force_target_matches = false;
        in.contact_force_magnitude = 0.0;
      }
    }
    in.primitives.clear();
    in.primitives.reserve(obj.primitives.size());
    for (const auto& prim : obj.primitives) {
      AttachedPrimitiveInput pin;
      // Unknown shape tag or too few dimensions → fail closed (an unrecognised
      // payload is not safe to ignore).
      if (!decode_attached_primitive(prim, pin)) {
        fail_closed("undecodable_primitive");
        return;
      }
      // Moved, not copied: `AttachedPrimitiveInput` now owns a vertex vector,
      // and a copy duplicates up to kMaxTightHullVertices Vec3 per primitive
      // (~123 kB for a fully refined object) on every world-state message, in
      // a node whose whole contract is allocation discipline.
      in.primitives.push_back(std::move(pin));
    }
    attached_ingest_scratch_.push_back(std::move(in));
  }

  union_bound_geometry(*msg);
  const AttachIngestStatus status = ingest_attached_objects(
      attached_ingest_scratch_, collision_link_names_, attached_max_objects_,
      attached_max_primitives_, attached_max_touch_links_, attached_model_);
  if (status != AttachIngestStatus::kOk) {
    attached_overflow_ = true;
    attached_received_ = true;
    attached_stamp_ = producer_stamp;
    RCLCPP_WARN(this->get_logger(), "safety.attached_ingest_rejected status=%d n=%zu",
                static_cast<int>(status), wire.size());
    // A rejected payload model takes the region scoped to it: the object mask
    // the region was resolved against no longer describes anything.
    place_region_ = PlaceApproachRegion{};
    if (grasp_identity_armed()) {
      retire_grasp_declaration("attachment_rejected");
    }
    return;
  }
  // Labels and provenance parallel the accepted objects. Provenance is copied
  // only for diagnostic FailureTrigger evidence; it never reaches a check.
  for (std::size_t i = 0; i < attached_model_.n_objects; ++i) {
    attached_labels_[i] = wire[i].object_id;
    attached_evidence_kinds_[i] = wire[i].evidence_kind;
    attached_evidence_refs_[i] = wire[i].evidence_ref;
  }
  check_attachment_bindings(*msg);
  apply_tracked_release();
  // Arm a support-contact witness only when World State attests a genuinely NEW
  // one. The attachment set is heartbeated, so keying off message arrival would
  // silently resurrect an exemption the kernel had already killed on
  // separation. The key is (object id, support id, producer stamp): a regrasp
  // or a re-contact produces a fresh stamp and re-arms honestly; a republished
  // snapshot does not.
  for (std::size_t i = 0; i < support_witness_keys_.size(); ++i) {
    const auto bit = static_cast<std::uint8_t>(1U << i);
    // An off-hand tracked binding (#418 K3) arms no witness: a payload the
    // kernel cannot place in the hand licenses no contact.
    if (i >= attached_model_.n_objects || !attached_model_.objects[i].has_support_witness ||
        attached_binding_[i].off_hand != nullptr) {
      support_witness_keys_[i] = SupportWitnessKey{};
      support_witness_live_ = static_cast<std::uint8_t>(support_witness_live_ & ~bit);
      continue;
    }
    SupportWitnessKey key;
    key.object_id = wire[i].object_id;
    key.support_id = wire[i].support_contact.support_id;
    key.stamp_ns = wire[i].support_contact.stamp_ns;
    key.valid = true;
    if (!(key == support_witness_keys_[i])) {
      support_witness_keys_[i] = key;
      support_witness_live_ = static_cast<std::uint8_t>(support_witness_live_ | bit);
      RCLCPP_INFO(this->get_logger(), "safety.support_witness_armed object=%s support=%s",
                  key.object_id.c_str(), key.support_id.c_str());
    }
  }
  // Attachment-revision edge FIRST, and specifically before the declaration is
  // re-ingested: a detach retires the payload the region is scoped to, and a
  // region ingested against a payload that is already gone resolves to an empty
  // object mask — which the ingest can only report as a refusal. Round-8
  // (`spark:~/openral-runs/2026-08-15-round8/`) logged every clean release that
  // way, as `place_region_rejected`, when what happened was an ordinary disarm.
  // Announcing the drop here keeps a detach one `place_region_dropped` line.
  if (msg->attachment_revision != attached_revision_) {
    attached_revision_ = msg->attachment_revision;
    if (attached_model_.n_objects == 0) {
      attached_contact_snapshot_pending_ = false;
      attached_contact_active_ = false;
      // Detach: every exemption dies with the payload it belonged to.
      support_witness_live_ = 0;
      std::fill(support_witness_keys_.begin(), support_witness_keys_.end(), SupportWitnessKey{});
      std::fill(attached_contact_mask_.begin(), attached_contact_mask_.end(), 0);
      std::fill(attached_contact_distance_.begin(), attached_contact_distance_.end(),
                std::numeric_limits<double>::infinity());
      // Detach: the approach allowance dies with the payload it was scoped to,
      // exactly as the witness does.
      if (place_region_.valid) {
        RCLCPP_INFO(this->get_logger(), "safety.place_region_dropped reason=detached target=%s",
                    place_declaration_target_.c_str());
      }
      place_region_ = PlaceApproachRegion{};
      attached_model_.force_gate = PlaceForceGate{};
      place_region_refusal_reason_.clear();
      place_region_refusal_target_.clear();
      // A detach while a grasp region is armed retires it, before or after its
      // handover: either the grasped payload was released (the handover is over)
      // or something else let go — another hand, a frozen release record — and
      // the scene the region was measured in is gone. Retirement is decided on
      // the revision edge alone, never on which hand the kernel last saw holding
      // (with KeepLast(1) a missed ATTACH snapshot would make the declaring
      // hand's own release look like another's), and a retired identity never
      // re-arms. Liveness is the producer's: it sees every DETACH, drops its
      // other hand's pre-handover region and re-arms under a fresh identity from
      // a measurement taken after the detach.
      if (grasp_identity_armed()) {
        retire_grasp_declaration("detached");
      }
    } else {
      attached_contact_snapshot_pending_ = true;
      attached_contact_active_ = false;
    }
  }
  // The place declaration rides the same snapshot as the payload it is scoped
  // to, so it is resolved here, against the objects that were just accepted.
  ingest_place_declaration(*msg);
  if (grasp_allowance_enabled_) {
    ingest_grasp_declaration(*msg);
  }
  if (tracked_grasp_enabled_ || provider_ != nullptr) {
    ingest_tracked_declaration_key(*msg);
  }
  if (workspace_ingest_enabled_) {
    ingest_workspace_contact(*msg);
  }
  attached_overflow_ = false;
  attached_received_ = true;
  attached_stamp_ = producer_stamp;
  if (attachment_applied_pub_ != nullptr) {
    std_msgs::msg::UInt64 applied;
    applied.data = msg->attachment_revision;
    attachment_applied_pub_->publish(applied);
  }
}

void SafetyKernelLifecycleNode::ingest_place_declaration(
    const openral_msgs::msg::WorldStateStamped& msg) {
  const bool was_valid = place_region_.valid;
  const std::size_t was_geometry = place_region_.n_geometry;
  const std::string previous_target = place_declaration_target_;
  place_region_ = PlaceApproachRegion{};
  // ADR-0100: the force gate is re-derived from scratch on every snapshot, for
  // the same reason the region is — a bound that outlived its declaration is an
  // armed enforcement surface nobody declared.
  attached_model_.force_gate = PlaceForceGate{};
  place_declaration_stamp_ns_ = 0;
  place_declaration_timeout_s_ = 0.0;
  place_declaration_target_.clear();
  place_region_stamp_ns_ = 0;

  const auto announce_dropped = [&](const char* reason) {
    // A declaration that is gone is not a declaration being refused: the next
    // refusal, whatever it is, is news again.
    place_region_refusal_reason_.clear();
    place_region_refusal_target_.clear();
    if (was_valid) {
      RCLCPP_INFO(this->get_logger(), "safety.place_region_dropped reason=%s target=%s", reason,
                  previous_target.c_str());
    }
  };
  // The attachment set is heartbeated at 30 Hz and a refusal almost always
  // describes a STANDING state, not an event: a declaration published before the
  // grasp lands resolves to no carried payload on every beat of the approach.
  // Round-8 logged 672-811 such lines per run. Emit on the (reason, target)
  // transition only — the same discipline the support-contact witness logs are
  // under — and leave the standing state to the 1 Hz diagnostics heartbeat's
  // `place_region` key, which is where a state that persists belongs.
  const auto refusal_is_new = [this](const char* reason, const std::string& target) {
    if (place_region_refusal_reason_ == reason && place_region_refusal_target_ == target) {
      return false;
    }
    place_region_refusal_reason_ = reason;
    place_region_refusal_target_ = target;
    return true;
  };
  if (!msg.place_declaration_valid) {
    announce_dropped("no_declaration");
    return;
  }
  const auto& declaration = msg.place_declaration;
  // Attributability first (HZ-0097-2 mitigation 1): a wrong declaration must be
  // reconstructible from the trace whether or not it ends up arming anything.
  place_declaration_target_ = declaration.target_id;
  if (!declaration.active) {
    announce_dropped("retracted");
    return;
  }
  if (declaration.stamp_ns <= 0 || !std::isfinite(declaration.timeout_s) ||
      declaration.timeout_s <= 0.0 || declaration.timeout_s > kMaxPlaceForceGateTimeoutS) {
    if (refusal_is_new("invalid_liveness", declaration.target_id)) {
      RCLCPP_WARN(this->get_logger(),
                  "safety.place_force_gate_rejected reason=invalid_liveness stamp_ns=%ld "
                  "timeout_s=%g target=%s",
                  static_cast<long>(declaration.stamp_ns), declaration.timeout_s,
                  declaration.target_id.c_str());
    }
    announce_dropped("invalid_liveness");
    return;
  }
  place_declaration_stamp_ns_ = declaration.stamp_ns;
  place_declaration_timeout_s_ = declaration.timeout_s;
  // Which carried payload the declaration is scoped to. An empty object_id is
  // the direct-dispatch case ("whichever payload is carried"), which the
  // producer resolves at attach time; the kernel maps it onto every accepted
  // object. Resolved HERE, above the region checks, because the ADR-0100 force
  // gate travels with the *declaration* and must arm for the region-less
  // declaration dispatch actually publishes — unlike the approach allowance,
  // which needs a producer-measured region to mean anything.
  std::uint8_t object_mask = 0;
  const std::size_t objects = std::min<std::size_t>(attached_model_.n_objects, 8);
  for (std::size_t i = 0; i < objects; ++i) {
    if (declaration.object_id.empty() || attached_labels_[i] == declaration.object_id) {
      object_mask = static_cast<std::uint8_t>(object_mask | (1U << i));
    }
  }
  // ADR-0100. `<= 0` is the default and means no gate at all; a threshold past
  // the shared ceiling is a producer error and buys nothing, failing closed
  // toward the shipped geometry-only behaviour rather than toward a clamped
  // bound nobody chose.
  const double threshold_n = declaration.contact_force_threshold_n;
  if (object_mask != 0 && std::isfinite(threshold_n) && threshold_n > 0.0 &&
      threshold_n <= kMaxContactForceThresholdN) {
    attached_model_.force_gate.valid = true;
    attached_model_.force_gate.object_mask = object_mask;
    attached_model_.force_gate.threshold_n = threshold_n;
    attached_model_.force_gate.declaration_stamp_ns = declaration.stamp_ns;
    attached_model_.force_gate.declaration_timeout_s = declaration.timeout_s;
    attached_model_.force_gate.witness_max_age_s = attached_collision_deadline_s_;
  } else if (threshold_n > kMaxContactForceThresholdN &&
             refusal_is_new("force_threshold_out_of_range", declaration.target_id)) {
    RCLCPP_WARN(this->get_logger(),
                "safety.place_force_gate_rejected reason=out_of_range threshold_n=%g ceiling_n=%g "
                "target=%s",
                threshold_n, kMaxContactForceThresholdN, declaration.target_id.c_str());
  }
  if (!declaration.region_valid) {
    // The common, correct case for dispatch's own publication and for every real
    // deployment today: a declaration with no producer-measured region. It still
    // gates the place witness; it buys no approach allowance.
    announce_dropped("no_region");
    return;
  }
  const auto& region = declaration.region;
  if (!region_measurement_fresh(region.stamp_ns, place_region_max_age_s_)) {
    // A measurement older than the bound (or from the future) is not evidence
    // of where the receptacle is now: no allowance, the undeclared margins.
    if (was_valid) {
      RCLCPP_INFO(this->get_logger(), "safety.place_region_dropped reason=region_stale target=%s",
                  previous_target.c_str());
    }
    if (refusal_is_new("region_stale", declaration.target_id)) {
      RCLCPP_WARN(this->get_logger(),
                  "safety.place_region_rejected reason=region_stale target=%s age_s=%g "
                  "max_age_s=%g rskill=%s trace=%s evidence=%s",
                  declaration.target_id.c_str(),
                  static_cast<double>(this->now().nanoseconds() - region.stamp_ns) * 1e-9,
                  place_region_max_age_s_, declaration.rskill_id.c_str(),
                  declaration.trace_id.c_str(), region.evidence_ref.c_str());
    }
    return;
  }
  if (voxel_frame_id_.empty() || region.frame_id != voxel_frame_id_) {
    if (refusal_is_new("frame_mismatch", declaration.target_id)) {
      RCLCPP_WARN(
          this->get_logger(),
          "safety.place_region_rejected reason=frame_mismatch region_frame=%s grid_frame=%s "
          "target=%s",
          region.frame_id.c_str(), voxel_frame_id_.c_str(), declaration.target_id.c_str());
    }
    return;
  }
  const Transform pose = transform_from_translation_quat(
      region.pose.position.x, region.pose.position.y, region.pose.position.z,
      region.pose.orientation.x, region.pose.orientation.y, region.pose.orientation.z,
      region.pose.orientation.w);
  const Vec3 half{region.half_extents.x, region.half_extents.y, region.half_extents.z};
  PlaceRegionStatus status = PlaceRegionStatus::kBadPose;
  if (is_unit_quaternion(region.pose.orientation)) {
    status = ingest_place_region(pose, half, object_mask, place_region_);
  } else {
    place_region_ = PlaceApproachRegion{};
  }
  if (status == PlaceRegionStatus::kOk) {
    // ADR-0098: the declared target's own geometry, decoded off the same
    // message and validated the same fail-closed way. A producer that names a
    // target and then describes it with a shape the kernel cannot measure is
    // not one whose BOX is trusted either — so a bad list takes the whole
    // region with it (#142/#146), leaving exactly the undeclared margins rather
    // than a silent downgrade to box-only that would read on the wire like a
    // producer that measured nothing.
    place_geometry_scratch_.clear();
    for (const auto& prim : region.geometry) {
      AttachedPrimitiveInput pin;
      if (!decode_attached_primitive(prim, pin)) {
        place_geometry_scratch_.clear();
        status = PlaceRegionStatus::kBadGeometry;
        break;
      }
      if (place_geometry_scratch_.size() >= kMaxPlaceTargetPrimitives) {
        place_geometry_scratch_.clear();
        status = PlaceRegionStatus::kGeometryOverflow;
        break;
      }
      place_geometry_scratch_.push_back(pin);
    }
    if (status == PlaceRegionStatus::kOk) {
      status =
          ingest_place_target_geometry(place_geometry_scratch_, place_geometry_, place_region_);
    }
    if (status != PlaceRegionStatus::kOk) {
      place_region_ = PlaceApproachRegion{};
    }
  }
  if (status != PlaceRegionStatus::kOk) {
    const char* reason = place_region_status_reason(status);
    if (refusal_is_new(reason, declaration.target_id)) {
      if (status == PlaceRegionStatus::kNoObject) {
        // Not a fault, and not the producer's: dispatch declares the place phase
        // for the whole goal, and the goal starts before the grasp — so until
        // the payload is attached the declaration names an object the kernel is
        // not carrying. Nothing is refused that would otherwise have been
        // granted, and the margins in force are exactly the undeclared ones, so
        // this is a state note at INFO rather than a warning. It is still
        // logged: an allowance that never armed has to be reconstructible from
        // the trace (HZ-0097-2 mitigation 1, CLAUDE.md §1.4).
        RCLCPP_INFO(this->get_logger(),
                    "safety.place_region_not_armed reason=no_object target=%s object=%s "
                    "attached=%zu rskill=%s trace=%s",
                    declaration.target_id.c_str(), declaration.object_id.c_str(),
                    attached_model_.n_objects, declaration.rskill_id.c_str(),
                    declaration.trace_id.c_str());
      } else {
        // Everything else is a malformed region in a message that reached the
        // kernel: a producer error, warned once per (reason, target) transition.
        RCLCPP_WARN(this->get_logger(),
                    "safety.place_region_rejected reason=%s target=%s object=%s half_m=%g,%g,%g "
                    "rskill=%s trace=%s",
                    reason, declaration.target_id.c_str(), declaration.object_id.c_str(), half.x,
                    half.y, half.z, declaration.rskill_id.c_str(), declaration.trace_id.c_str());
      }
    }
    return;
  }
  place_region_refusal_reason_.clear();
  place_region_refusal_target_.clear();
  place_region_stamp_ns_ = region.stamp_ns;
  // Announce on the arming transition, and again whenever the target's geometry
  // count changes: a region that gains or loses the declared body's primitives
  // is adjudicating against something materially different, and a producer whose
  // measurement started failing mid-goal would otherwise be silent (CLAUDE.md
  // §1.4). The count is stable across the 30 Hz heartbeat, so this stays a
  // transition line rather than a stream; the standing state is on the 1 Hz
  // `/diagnostics` `place_region` key.
  if (!was_valid || was_geometry != place_region_.n_geometry) {
    RCLCPP_INFO(this->get_logger(),
                "safety.place_region_armed target=%s object_mask=0x%x half_m=%g,%g,%g "
                "allowance_m=%g geometry=%zu rskill=%s trace=%s",
                declaration.target_id.c_str(), static_cast<unsigned>(object_mask), half.x, half.y,
                half.z, place_approach_allowance_cap(voxel_grid_.resolution),
                place_region_.n_geometry, declaration.rskill_id.c_str(),
                declaration.trace_id.c_str());
  }
}

bool SafetyKernelLifecycleNode::place_declaration_live() const noexcept {
  if (!place_region_.valid || place_declaration_timeout_s_ <= 0.0) {
    return false;
  }
  // Fails toward dead, the same three ways PlaceDeclaration::is_live does: past
  // the backstop, and stamped in the future (a clock that jumped is not evidence
  // of anything). The producer stopping publication is covered separately by the
  // attachment freshness deadline.
  const std::int64_t elapsed_ns = this->now().nanoseconds() - place_declaration_stamp_ns_;
  return elapsed_ns >= 0 &&
         elapsed_ns <= static_cast<std::int64_t>(place_declaration_timeout_s_ * 1e9) &&
         // The measurement ages between world-state messages too.
         region_measurement_fresh(place_region_stamp_ns_, place_region_max_age_s_);
}

bool SafetyKernelLifecycleNode::region_measurement_fresh(std::int64_t stamp_ns,
                                                         double max_age_s) const noexcept {
  if (!(max_age_s > 0.0)) {
    return false;
  }
  const std::int64_t age_ns = this->now().nanoseconds() - stamp_ns;
  return age_ns >= 0 && age_ns <= static_cast<std::int64_t>(max_age_s * 1e9);
}

void SafetyKernelLifecycleNode::ingest_tracked_objects(
    const openral_msgs::msg::WorldStateStamped& msg) {
  tracked_n_ = 0;  // every path but acceptance leaves the whole set ineligible
  if (!msg.tracked_objects_valid) {
    tracked_refusal_reason_.clear();  // no tracker publishing: nothing to refuse
    return;
  }
  const auto& set = msg.tracked_objects;
  // The objects count (<= 32) and each object's hull count (<= 4) are bounded
  // by the IDL sequences themselves; everything else is checked here.
  const char* reason = nullptr;
  if (voxel_frame_id_.empty() || set.frame_id != voxel_frame_id_) {
    reason = "frame_mismatch";
  } else if (tracked_have_revision_ && set.revision < tracked_revision_) {
    // Equal is the aggregator re-sending the same set on every beat.
    reason = "revision_regressed";
  } else if (set.stamp_ns > this->now().nanoseconds()) {
    reason = "future_stamp";
  }
  for (std::size_t i = 0; reason == nullptr && i < set.objects.size(); ++i) {
    const auto& obj = set.objects[i];
    if (obj.box.frame_id != set.frame_id) {
      reason = "frame_mismatch";
      break;
    }
    if (!obj.box.geometry.empty()) {
      reason = "geometry_not_supported";
      break;
    }
    for (const auto& hull : obj.tight) {
      const std::size_t values = hull.tight_hull_vertices.size();
      if (values % 3 != 0 || values / 3 > kMaxTrackedHullVertices) {
        reason = "hull_too_many_vertices";
        break;
      }
    }
    for (std::size_t k = 0; reason == nullptr && k < i; ++k) {
      if (set.objects[k].object_id == obj.object_id) {
        reason = "duplicate_object_id";
      }
    }
    if (reason != nullptr) {
      break;
    }
    // The grasp-region caps (half-extent, volume, finite, non-degenerate) are
    // the tracked box's caps; the region's link mask is unused here.
    TrackedEntry& entry = tracked_[i];
    const GraspRegionStatus status =
        is_unit_quaternion(obj.box.pose.orientation)
            ? ingest_grasp_region(
                  transform_from_translation_quat(
                      obj.box.pose.position.x, obj.box.pose.position.y, obj.box.pose.position.z,
                      obj.box.pose.orientation.x, obj.box.pose.orientation.y,
                      obj.box.pose.orientation.z, obj.box.pose.orientation.w),
                  Vec3{obj.box.half_extents.x, obj.box.half_extents.y, obj.box.half_extents.z},
                  std::bitset<kMaxGraspMaskLinks>{}.set(0), entry.box)
            : GraspRegionStatus::kBadPose;
    if (status != GraspRegionStatus::kOk) {
      reason = grasp_region_status_reason(status);
      break;
    }
    entry.object_id = obj.object_id;
    entry.epoch = obj.epoch;
    entry.live = obj.status == "confirmed" || obj.status == "occluded" || obj.status == "bound";
    entry.released = obj.status == "released";
    entry.target = obj.role == "target";
    entry.contact = entry.target || obj.role == "destination";
    entry.exemptible = obj.status == "confirmed" || obj.status == "occluded";
    entry.provisional = obj.provisional;
    entry.declaration_key = obj.declaration_key;
    entry.id_hash = std::hash<std::string>{}(obj.object_id);
    entry.confirmed_ns = obj.last_confirmed_stamp_ns;
    entry.visible_cells = obj.visible_cells;
    entry.cover_ok = false;
    entry.support_z = obj.support_z;  // K7; non-finite = no measured support
    entry.floor_z = obj.floor_z;      // OT-10 amendment; non-finite = none
  }
  if (reason != nullptr) {
    if (tracked_refusal_reason_ != reason) {
      tracked_refusal_reason_ = reason;
      RCLCPP_WARN(this->get_logger(),
                  "safety.tracked_objects_rejected reason=%s revision=%llu frame=%s "
                  "grid_frame=%s objects=%zu: every tracked object is ineligible",
                  reason, static_cast<unsigned long long>(set.revision), set.frame_id.c_str(),
                  voxel_frame_id_.c_str(), set.objects.size());
    }
    return;
  }
  tracked_refusal_reason_.clear();
  tracked_n_ = set.objects.size();
  tracked_set_frame_ = set.frame_id;
  tracked_cover_dirty_ = true;
  tracked_revision_ = set.revision;
  tracked_have_revision_ = true;
}

void SafetyKernelLifecycleNode::check_attachment_bindings(
    const openral_msgs::msg::WorldStateStamped& msg) {
  bool fk_done = false;
  for (auto& f : pinch_floor_) {
    f.seen = false;
  }
  for (std::size_t i = 0; i < attached_model_.n_objects; ++i) {
    const auto& wire = msg.attached_objects[i];
    AttachedBinding& binding = attached_binding_[i];
    binding = AttachedBinding{};
    binding.epoch = wire.epoch;
    binding.tracked = wire.evidence_kind == "tracked_object";
    const auto ensure_fk = [&]() {
      if (!fk_done) {
        // The MEASURED configuration, base dofs zeroed exactly as the candidate
        // path's FK does, into the binding's own scratch.
        std::copy(q_meas_.begin(), q_meas_.end(), q_fk_.begin());
        for (const int d : collision_base_dofs_) {
          q_fk_[static_cast<std::size_t>(d)] = 0.0;
        }
        forward_kinematics(collision_model_, q_fk_.data(), q_fk_.size(), binding_scratch_);
        fk_done = true;
      }
    };
    if (!binding.tracked) {
      // Legacy and PINCH_FALLBACK payloads: today's rules, unchanged. The one thing a fallback
      // payload may carry is the measured box of EXACTLY ONE provisional candidate it names
      // (`provisional:<id>`); that candidate, if live, settled and at the hand, is its own track
      // for the payload-own-cell exemption (OT-10 amendment). Several or ambiguous candidates
      // may be neighbours: no own track, no exemption.
      if (payload_own_enabled_ && wire.evidence_kind == "pinch_fallback" &&
          wire.evidence_ref.rfind("provisional:", 0) == 0 &&
          wire.evidence_ref.find(',') == std::string::npos && measured_state_fresh()) {
        const std::string id = wire.evidence_ref.substr(std::string("provisional:").size());
        for (std::size_t k = 0; k < tracked_n_; ++k) {
          const TrackedEntry& e = tracked_[k];
          if (!e.live || !e.provisional || e.object_id != id) {
            continue;
          }
          ensure_fk();
          const double cell = voxel_grid_.resolution > 0.0 ? voxel_grid_.resolution : 0.0;
          if (tracked_box_meets_hand(collision_model_, binding_scratch_, attached_model_,
                                     attached_model_.objects[i], e.box.pose, e.box.half_extents,
                                     cell) &&
              !tracked_retired_.contains(e.id_hash, e.epoch) && std::isfinite(e.floor_z)) {
            binding.own_armed = true;
            binding.own_track = static_cast<int>(k);
            binding.own_floor_z = e.floor_z;
          }
          break;
        }
      } else if (pinch_own_enabled_ && wire.evidence_kind == "pinch_fallback" &&
                 wire.evidence_ref == "untracked_grasp" && measured_state_fresh()) {
        // An untracked hold: the pinch volume is the object in the hand. Its floor is the
        // box's lowest point the first time this hold is seen (the fingers closed above the
        // support), kept while the hold lasts (Spark R191 / R194 / R200 / R203 / R206).
        PinchFloor* f = nullptr;
        for (auto& c : pinch_floor_) {
          if (c.used && c.object_id == wire.object_id && c.attach_link == wire.attach_link) {
            f = &c;
            break;
          }
        }
        if (f == nullptr) {
          for (auto& c : pinch_floor_) {
            if (!c.used) {
              const AttachedObject& obj = attached_model_.objects[i];
              if (obj.attach_link < 0 ||
                  static_cast<std::size_t>(obj.attach_link) >= collision_model_.n_links) {
                break;  // frozen or unknown: no hand, no hold
              }
              ensure_fk();
              const Transform object_xf = rigid_compose(
                  binding_scratch_.link_world[static_cast<std::size_t>(obj.attach_link)],
                  obj.pose_in_link);
              double low = std::numeric_limits<double>::infinity();
              for (int p = 0; p < obj.prim_count; ++p) {
                const AttachedPrimitive& prim =
                    attached_model_.primitives[static_cast<std::size_t>(obj.prim_first + p)];
                const Transform x = rigid_compose(object_xf, prim.pose_in_object);
                const double reach = std::abs(x.r[6]) * prim.half_extents.x +
                                     std::abs(x.r[7]) * prim.half_extents.y +
                                     std::abs(x.r[8]) * prim.half_extents.z;
                low = std::min(low, x.t.z - reach);
              }
              if (std::isfinite(low)) {
                c = PinchFloor{wire.object_id, wire.attach_link, low, true, false};
                f = &c;
                RCLCPP_INFO(this->get_logger(), "safety.pinch_own_floor object=%s floor_z=%.4f",
                            wire.object_id.c_str(), low);
              }
              break;
            }
          }
        }
        if (f != nullptr) {
          f->seen = true;
          binding.own_armed = true;
          binding.own_fallback = true;
          binding.own_floor_z = f->floor_z;
        }
      }
      continue;
    }
    const TrackedEntry* entry = nullptr;
    int entry_k = -1;
    for (std::size_t k = 0; k < tracked_n_; ++k) {
      if (tracked_[k].live && tracked_[k].epoch == wire.epoch &&
          tracked_[k].object_id == wire.object_id) {
        entry = &tracked_[k];
        entry_k = static_cast<int>(k);
        break;
      }
    }
    if (tracked_n_ == 0) {
      binding.off_hand = "no_accepted_set";
    } else if (entry == nullptr) {
      binding.off_hand = "no_live_track";
    } else if (entry->provisional) {
      // A provisional fit may under-contain the object: exemption geometry only,
      // never a payload binding (the tracker never binds one; fail closed anyway).
      binding.off_hand = "provisional_track";
    } else if (!measured_state_fresh()) {
      binding.off_hand = "no_measured_state";
    } else {
      ensure_fk();
      const double cell = voxel_grid_.resolution > 0.0 ? voxel_grid_.resolution : 0.0;
      if (!tracked_box_meets_hand(collision_model_, binding_scratch_, attached_model_,
                                  attached_model_.objects[i], entry->box.pose,
                                  entry->box.half_extents, cell)) {
        binding.off_hand = "box_off_hand";
      }
    }
    // OT-10 amendment: a K3-valid, unretired binding with a floor owns its measured cells.
    if (payload_own_enabled_ && binding.off_hand == nullptr && entry != nullptr &&
        !tracked_retired_.contains(entry->id_hash, entry->epoch) && std::isfinite(entry->floor_z)) {
      binding.own_armed = true;
      binding.own_track = entry_k;
      binding.own_floor_z = entry->floor_z;
    }
    if (binding.off_hand != nullptr) {
      const auto wall = std::chrono::steady_clock::now();
      if (wall - binding_last_warn_ >= std::chrono::seconds(1)) {
        binding_last_warn_ = wall;
        RCLCPP_WARN(this->get_logger(),
                    "safety.attachment_binding_off_hand object=%s epoch=%u reason=%s: still "
                    "checked as a payload; no handover, no support witness",
                    wire.object_id.c_str(), static_cast<unsigned>(wire.epoch), binding.off_hand);
      }
    }
  }
  for (auto& f : pinch_floor_) {
    if (f.used && !f.seen) {
      f = PinchFloor{};  // the hold ended: the next one is measured afresh
    }
  }
}

std::size_t SafetyKernelLifecycleNode::fill_payload_own() {
  PayloadOwnCells& t = payload_own_;
  t.valid = false;
  t.armed = 0;
  t.fallback_ok = 0;
  t.n_others = 0;
  t.own.fill(0);
  t.floor_z.fill(std::numeric_limits<double>::quiet_NaN());
  t.static_cells = nullptr;
  t.static_cells_n = 0;
  // Fail closed: a tracked set the kernel does not hold, one in another frame, a grid it cannot
  // cover-check or an attachment stream that is no longer fresh exempts nothing.
  const bool frame_ok = refresh_tracked_cover();
  if (!frame_ok || tracked_n_ == 0 || voxel_overflow_ || !tracked_stream_fresh()) {
    return 0;
  }
  std::array<int, kMaxTrackedObjects> slot{};
  slot.fill(-1);
  for (std::size_t k = 0; k < tracked_n_ && k < slot.size(); ++k) {
    const TrackedEntry& e = tracked_[k];
    // Every settled track is "another body" to a payload that is not its own: confirmed /
    // occluded / bound (provisional or not), never a retired identity.
    if (!e.live || tracked_retired_.contains(e.id_hash, e.epoch)) {
      continue;
    }
    slot[k] = static_cast<int>(t.n_others);
    t.other_box[t.n_others] = e.box.pose;
    t.other_half[t.n_others] = e.box.half_extents;
    ++t.n_others;
  }
  std::size_t armed = 0;
  for (std::size_t i = 0; i < attached_model_.n_objects && i < 8; ++i) {
    const AttachedBinding& b = attached_binding_[i];
    if (b.own_armed && b.own_fallback && std::isfinite(b.own_floor_z)) {
      t.armed = static_cast<std::uint8_t>(t.armed | (1U << i));
      t.fallback_ok = static_cast<std::uint8_t>(t.fallback_ok | (1U << i));
      t.floor_z[i] = b.own_floor_z;
      ++armed;
      continue;
    }
    if (!b.own_armed || b.own_track < 0 || static_cast<std::size_t>(b.own_track) >= slot.size() ||
        slot[static_cast<std::size_t>(b.own_track)] < 0) {
      continue;
    }
    t.armed = static_cast<std::uint8_t>(t.armed | (1U << i));
    t.floor_z[i] = b.own_floor_z;
    t.own[i] |= 1U << static_cast<unsigned>(slot[static_cast<std::size_t>(b.own_track)]);
    ++armed;
  }
  const bool static_ok = static_cells_marked_ > 0 && !voxel_overflow_;
  if (!static_points_.empty() && !static_ok && armed > 0) {
    // Fail closed (OT-10): declared static colliders this grid's table cannot name (off the base
    // lattice, or no point marks a cell) would leave rule 3 unchecked, so exempt nothing.
    t.armed = 0;
    t.fallback_ok = 0;
    armed = 0;
    const char* why = !static_cells_for_valid_    ? "pending"
                      : !static_cells_lattice_ok_ ? "lattice"
                                                  : "no_cell";
    if (why != payload_own_disarm_logged_) {  // once per change of reason, per node
      payload_own_disarm_logged_ = why;
      RCLCPP_WARN(this->get_logger(),
                  "safety.payload_own_disarmed reason=%s: %zu declared static points mark no "
                  "cell of this grid, so no payload cell is exempt",
                  why, static_points_.size());
    }
  } else if (armed > 0) {
    payload_own_disarm_logged_ = nullptr;
  }
  if (static_ok) {
    t.static_cells = static_cells_.data();
    t.static_cells_n = static_cast<std::size_t>(voxel_grid_.sx) *
                       static_cast<std::size_t>(voxel_grid_.sy) *
                       static_cast<std::size_t>(voxel_grid_.sz);
  }
  t.valid = armed > 0;
  if (armed != payload_own_logged_n_) {
    payload_own_logged_n_ = armed;
    RCLCPP_INFO(this->get_logger(),
                "safety.payload_own_armed objects=%zu of=%zu others=%zu static_cells=%zu", armed,
                attached_model_.n_objects, t.n_others,
                static_ok ? static_cells_marked_ : std::size_t{0});
  }
  return armed;
}

void SafetyKernelLifecycleNode::union_bound_geometry(
    const openral_msgs::msg::WorldStateStamped& msg) {
  for (auto& e : bound_) {
    e.seen = false;
  }
  std::fill(bound_slot_.begin(), bound_slot_.end(), -1);
  for (std::size_t i = 0; i < attached_ingest_scratch_.size(); ++i) {
    const auto& wire = msg.attached_objects[i];
    if (wire.evidence_kind != "tracked_object") {
      continue;  // legacy and PINCH_FALLBACK payloads: replaced per message, as today
    }
    AttachedObjectInput& in = attached_ingest_scratch_[i];
    BoundPayload* e = nullptr;
    for (auto& c : bound_) {
      if (c.used && c.object_id == wire.object_id && c.attach_link == wire.attach_link) {
        e = &c;
        break;
      }
    }
    if (e != nullptr && e->epoch != wire.epoch) {
      e->used = false;  // a new epoch starts fresh
    }
    if (e == nullptr) {
      for (auto& c : bound_) {
        if (!c.used || !c.seen) {
          // A slot another binding left (off this wire) starts fresh: kept `used`, it handed
          // that binding's K4 union and K5 release to this one (Spark R285, 2026-10-23: the
          // orange just bound inherited the released one's closed window and frozen box, and
          // the gripper stopped against it).
          c.used = false;
          e = &c;
          break;
        }
      }
    }
    if (e == nullptr) {
      continue;  // unreachable: more tracked payloads than slots; today's semantics
    }
    if (!e->used) {
      if (in.primitives.size() > kMaxBoundPrimitives) {
        RCLCPP_WARN(this->get_logger(),
                    "safety.tracked_geometry_cap object=%s epoch=%u primitives=%zu cap=%zu: "
                    "checked as published, no union held",
                    wire.object_id.c_str(), static_cast<unsigned>(wire.epoch), in.primitives.size(),
                    kMaxBoundPrimitives);
        continue;
      }
      *e = BoundPayload{};  // K4 and K5 state both start fresh
      e->used = true;
      e->object_id = wire.object_id;
      e->attach_link = wire.attach_link;
      e->epoch = wire.epoch;
      e->revision = wire.geometry_revision;
      e->frame = in.pose_in_link;
      e->n = in.primitives.size();
      std::copy(in.primitives.begin(), in.primitives.end(), e->prims.begin());
    }
    e->seen = true;
    if (i < bound_slot_.size()) {
      bound_slot_[i] = static_cast<int>(e - bound_.data());
    }
    // This message's object frame, expressed in the union's.
    const Transform to_union = rigid_compose(rigid_inverse(e->frame), in.pose_in_link);
    {
      // Any primitive not yet held is unioned in. A lower or equal revision
      // carrying new content is a producer that changed geometry without
      // bumping it: still unioned (growth is the safe direction), and WARNed.
      std::size_t n = e->n;
      std::array<AttachedPrimitiveInput, kMaxBoundPrimitives> grown = e->prims;
      bool over = false;
      for (const auto& p : in.primitives) {
        AttachedPrimitiveInput q = p;
        q.pose_in_object = rigid_compose(to_union, p.pose_in_object);
        const auto held_same =
            std::find_if(grown.begin(), grown.begin() + static_cast<std::ptrdiff_t>(n),
                         [&q](const AttachedPrimitiveInput& h) { return same_primitive(h, q); });
        if (held_same != grown.begin() + static_cast<std::ptrdiff_t>(n)) {
          // The same geometry published again keeps its place in the union; a publisher that
          // later says it is only fallback downgrades it (the evidence can only get weaker).
          if (q.source == PrimitiveSource::kFallback) {
            held_same->source = PrimitiveSource::kFallback;
          }
          continue;
        }
        if (n == kMaxBoundPrimitives) {
          over = true;
          break;
        }
        grown[n++] = std::move(q);
      }
      if (over) {
        const auto wall = std::chrono::steady_clock::now();
        if (wall - bound_cap_last_warn_ >= std::chrono::seconds(1)) {
          bound_cap_last_warn_ = wall;
          RCLCPP_WARN(this->get_logger(),
                      "safety.tracked_geometry_cap object=%s epoch=%u revision=%llu cap=%zu: "
                      "update refused, the held union (revision %llu) stays",
                      wire.object_id.c_str(), static_cast<unsigned>(wire.epoch),
                      static_cast<unsigned long long>(wire.geometry_revision), kMaxBoundPrimitives,
                      static_cast<unsigned long long>(e->revision));
        }
      } else if (n > e->n) {
        if (wire.geometry_revision <= e->revision) {
          const auto wall = std::chrono::steady_clock::now();
          if (wall - bound_cap_last_warn_ >= std::chrono::seconds(1)) {
            bound_cap_last_warn_ = wall;
            RCLCPP_WARN(this->get_logger(),
                        "safety.tracked_geometry_unrevised object=%s epoch=%u revision=%llu "
                        "held=%llu: new geometry without a higher revision, unioned anyway",
                        wire.object_id.c_str(), static_cast<unsigned>(wire.epoch),
                        static_cast<unsigned long long>(wire.geometry_revision),
                        static_cast<unsigned long long>(e->revision));
          }
        }
        e->prims = std::move(grown);
        e->n = n;
      }
      e->revision = std::max(e->revision, wire.geometry_revision);
    }
    // The support witness rides this message's object frame: re-express it.
    if (in.has_support_witness) {
      Transform point;
      point.t = in.support_point;
      in.support_point = rigid_compose(to_union, point).t;
      Transform normal;
      normal.t = in.support_normal;
      Transform rot = to_union;
      rot.t = Vec3{};
      in.support_normal = rigid_compose(rot, normal).t;
    }
    in.pose_in_link = e->frame;
    in.primitives.assign(e->prims.begin(), e->prims.begin() + static_cast<std::ptrdiff_t>(e->n));
  }
}

void SafetyKernelLifecycleNode::apply_tracked_release() {
  if (release_jaw_cols_.empty()) {
    return;
  }
  const std::int64_t now = this->now().nanoseconds();
  const bool fresh = measured_state_fresh();
  bool fk_done = false;
  const auto fk = [&]() {
    if (!fk_done) {
      std::copy(q_meas_.begin(), q_meas_.end(), q_fk_.begin());
      for (const int d : collision_base_dofs_) {
        q_fk_[static_cast<std::size_t>(d)] = 0.0;
      }
      forward_kinematics(collision_model_, q_fk_.data(), q_fk_.size(), binding_scratch_);
      fk_done = true;
    }
  };
  for (std::size_t i = 0; i < attached_model_.n_objects && i < bound_slot_.size(); ++i) {
    if (bound_slot_[i] < 0) {
      continue;
    }
    BoundPayload& e = bound_[static_cast<std::size_t>(bound_slot_[i])];
    AttachedObject& obj = attached_model_.objects[i];
    if (e.release == BoundPayload::Release::kHeld) {
      // The tracker letting go of this (id, epoch) releases it here too: one release
      // decision for both sides (2026-10-22 decision; Spark R138: the tracker's jaw rule
      // fired, this one did not, and the payload rode the hand after the banana fell).
      // Measured state is needed to freeze it where it was let go. Safety-WG review owed.
      bool by_tracker = false;
      for (std::size_t k = 0; fresh && k < tracked_n_; ++k) {
        by_tracker = by_tracker || (tracked_[k].released && tracked_[k].epoch == e.epoch &&
                                    tracked_[k].object_id == e.object_id);
      }
      if (by_tracker) {
        fk();
        e.frozen =
            rigid_compose(binding_scratch_.link_world[static_cast<std::size_t>(obj.attach_link)],
                          obj.pose_in_link);
        e.hand_link = obj.attach_link;
        e.release = BoundPayload::Release::kWindow;
        e.window_opened_ns = now;
        RCLCPP_WARN(this->get_logger(),
                    "safety.tracked_release object=%s epoch=%u by=tracker: frozen in base, hand "
                    "and touch links exempt until %g m clear or %g s (#418 K5, not monotone)",
                    e.object_id.c_str(), static_cast<unsigned>(e.epoch), release_clear_m_,
                    release_timeout_s_);
        if (grasp_latched_ && grasp_latched_tracked_ && grasp_latched_object_ == e.object_id &&
            grasp_latched_epoch_ == e.epoch) {
          retire_grasp_declaration("tracked_release");
        }
      }
    }
    if (e.release == BoundPayload::Release::kHeld) {
      // The jaw driving the attach link or a touch link (a mimic follower reads
      // its leader's column).
      for (std::size_t k = 0; e.jaw < 0 && k < release_jaw_cols_.size(); ++k) {
        for (int t = -1; t < obj.touch_count; ++t) {
          const int link =
              t < 0 ? obj.attach_link
                    : attached_model_.touch_links[static_cast<std::size_t>(obj.touch_first + t)];
          if (link >= 0 &&
              collision_model_.dof_index[static_cast<std::size_t>(link)] == release_jaw_cols_[k]) {
            e.jaw = static_cast<int>(k);
            break;
          }
        }
      }
      if (e.jaw < 0 || !fresh || attached_binding_[i].off_hand != nullptr) {
        e.open_since_ns = -1;  // no jaw, no measurement or off the hand: today's rules
        continue;
      }
      const auto k = static_cast<std::size_t>(e.jaw);
      const double q = q_meas_[static_cast<std::size_t>(release_jaw_cols_[k])];
      if (!e.hold_set) {
        e.hold = q;
        e.hold_set = true;
        continue;
      }
      if (!(release_jaw_sign_[k] * (q - e.hold) > release_gap_)) {
        e.open_since_ns = -1;
        continue;
      }
      if (e.open_since_ns < 0) {
        e.open_since_ns = now;
      }
      if (static_cast<double>(now - e.open_since_ns) < release_dwell_s_ * 1e9) {
        continue;
      }
      fk();
      e.frozen = rigid_compose(
          binding_scratch_.link_world[static_cast<std::size_t>(obj.attach_link)], obj.pose_in_link);
      e.hand_link = obj.attach_link;
      e.release = BoundPayload::Release::kWindow;
      e.window_opened_ns = now;
      RCLCPP_WARN(this->get_logger(),
                  "safety.tracked_release object=%s epoch=%u jaw=%.3f hold=%.3f: frozen in base, "
                  "hand and touch links exempt until %g m clear or %g s (#418 K5, not monotone)",
                  e.object_id.c_str(), static_cast<unsigned>(e.epoch), q, e.hold, release_clear_m_,
                  release_timeout_s_);
      if (grasp_latched_ && grasp_latched_tracked_ && grasp_latched_object_ == e.object_id &&
          grasp_latched_epoch_ == e.epoch) {
        retire_grasp_declaration("tracked_release");
      }
    }
    // Released: the payload stays where it was let go, whatever the wire says.
    obj.attach_link = -1;
    obj.pose_in_link = e.frozen;
    obj.has_support_witness = false;
    attached_binding_[i].off_hand = "released";  // never a handover, no witness
    if (e.release == BoundPayload::Release::kWindow) {
      obj.release_hand_link = e.hand_link;
      const char* why = nullptr;
      if (static_cast<double>(now - e.window_opened_ns) >= release_timeout_s_ * 1e9) {
        why = "timeout";
      } else if (fresh) {
        fk();
        if (released_payload_clearance(collision_model_, binding_scratch_, attached_model_, obj) >
            release_clear_m_) {
          why = "separation";
        }
      }
      if (why != nullptr) {
        e.release = BoundPayload::Release::kClosed;
        RCLCPP_INFO(this->get_logger(),
                    "safety.tracked_release_closed object=%s epoch=%u reason=%s: a plain "
                    "obstacle from here",
                    e.object_id.c_str(), static_cast<unsigned>(e.epoch), why);
      }
    }
    if (e.release == BoundPayload::Release::kClosed) {
      obj.release_hand_link = -1;
      obj.touch_count = 0;  // nothing is exempt against it any more
      // The frozen box hands over to the map once the map carries the object: the
      // tracker no longer calls this (id, epoch) bound or released (the two statuses the
      // octomap bridge clears an identity's cells for) and the map has cells inside the box.
      // Latched: a hand occluding it on a regrasp must not bring a stale box back around the
      // fingers (Spark R263, 2026-10-22: the orange's frozen box blocked the regrasp for 16 s).
      // Neither, and the box stays. Safety-WG review owed.
      if (!e.in_map && tracked_n_ > 0 && tracked_stream_fresh()) {
        bool cleared = false;
        for (std::size_t k = 0; k < tracked_n_; ++k) {
          const bool bound = tracked_[k].live && !tracked_[k].exemptible;
          cleared = cleared || ((tracked_[k].released || bound) && tracked_[k].epoch == e.epoch &&
                                tracked_[k].object_id == e.object_id);
        }
        std::size_t cover = 0;
        for (int p = 0; !cleared && p < obj.prim_count; ++p) {
          const AttachedPrimitive& prim =
              attached_model_.primitives[static_cast<std::size_t>(obj.prim_first + p)];
          const Vec3 half = prim.kind == AttachedShapeKind::kBox
                                ? prim.half_extents
                                : Vec3{prim.radius, prim.radius, prim.radius + prim.half_length};
          cover +=
              tracked_box_cover(voxel_grid_, rigid_compose(e.frozen, prim.pose_in_object), half);
        }
        if (!cleared && tracked_cover_sufficient(cover, 0)) {
          e.in_map = true;
          RCLCPP_INFO(this->get_logger(),
                      "safety.tracked_release_in_map object=%s epoch=%u cells=%zu: the map "
                      "carries it, the frozen box is dropped",
                      e.object_id.c_str(), static_cast<unsigned>(e.epoch), cover);
        }
      }
      if (e.in_map) {
        obj.prim_count = 0;
      }
    }
  }
}

bool SafetyKernelLifecycleNode::grasp_identity_armed() const noexcept {
  // The live region, or the current declaration's identity having armed before
  // and merely failing its last check (a stale measurement, a passing frame
  // refusal): an edge that ends the scene the region was measured in retires it
  // either way, so it cannot re-arm on the next fresh measurement (HZ-0115-3).
  return grasp_region_.valid ||
         (!grasp_declaration_target_.empty() && grasp_armed_target_ == grasp_declaration_target_ &&
          grasp_armed_stamp_ns_ == grasp_declaration_stamp_ns_);
}

void SafetyKernelLifecycleNode::retire_grasp_declaration(const char* reason) {
  if (grasp_region_.valid) {
    RCLCPP_INFO(this->get_logger(), "safety.grasp_region_dropped reason=%s target=%s", reason,
                grasp_declaration_target_.c_str());
  } else {
    RCLCPP_INFO(this->get_logger(), "safety.grasp_declaration_retired reason=%s target=%s", reason,
                grasp_declaration_target_.c_str());
  }
  grasp_region_ = GraspTargetRegion{};
  voxel_grid_.grasp_region = GraspTargetRegion{};
  // #418 K6: the latched handover's tracked identity retires with it (a no-op
  // when already held), so the same (id, epoch) is never a handover again.
  if (grasp_latched_ && grasp_latched_tracked_) {
    tracked_retired_.insert(std::hash<std::string>{}(grasp_latched_object_), grasp_latched_epoch_);
  }
  const bool evicted = grasp_retired_.insert(std::hash<std::string>{}(grasp_declaration_target_),
                                             grasp_declaration_stamp_ns_);
  if (evicted && !grasp_retired_overflow_logged_) {
    grasp_retired_overflow_logged_ = true;
    RCLCPP_WARN(this->get_logger(),
                "safety.grasp_retired_overflow capacity=%zu target=%s: the oldest retired "
                "identity was evicted (still bounded by its goal's timeout_s)",
                kGraspRetiredCapacity, grasp_declaration_target_.c_str());
  }
}

void SafetyKernelLifecycleNode::ingest_grasp_declaration(
    const openral_msgs::msg::WorldStateStamped& msg) {
  const bool was_valid = grasp_region_.valid;
  const bool was_handover = grasp_region_.handover;
  const std::string previous_target = grasp_declaration_target_;
  grasp_region_ = GraspTargetRegion{};
  grasp_declaration_stamp_ns_ = 0;
  grasp_declaration_timeout_s_ = 0.0;
  grasp_declaration_target_.clear();
  grasp_declaration_refused_ = false;

  const auto announce_dropped = [&](const char* reason) {
    grasp_region_refusal_reason_.clear();
    grasp_region_refusal_target_.clear();
    if (was_valid) {
      RCLCPP_INFO(this->get_logger(), "safety.grasp_region_dropped reason=%s target=%s", reason,
                  previous_target.c_str());
    }
  };
  if (!msg.grasp_declaration_valid) {
    announce_dropped("no_declaration");
    return;
  }
  const auto& declaration = msg.grasp_declaration;
  // Attributability first (HZ-0115-2): recorded before any arming decision, so
  // a declaration that never armed is still reconstructible from the trace.
  grasp_declaration_target_ = declaration.target_id;
  grasp_declaration_stamp_ns_ = declaration.stamp_ns;
  grasp_declaration_timeout_s_ = declaration.timeout_s;
  // Has this declaration (target, stamp) latched at a handover? Then its pick is
  // under way: the box is frozen (below), and the declaration losing its region
  // or being retracted ends the pick — it retires rather than merely dropping,
  // so the same identity can never re-arm against a later attachment (HZ-0115-3).
  const bool latched_declaration = grasp_latched_ &&
                                   grasp_latched_target_ == declaration.target_id &&
                                   grasp_latched_stamp_ns_ == declaration.stamp_ns;
  if (!declaration.active) {
    announce_dropped("retracted");
    if (latched_declaration) {
      retire_grasp_declaration("retracted");
    }
    return;
  }
  if (grasp_retired_.contains(std::hash<std::string>{}(declaration.target_id),
                              declaration.stamp_ns)) {
    // The heartbeat of a declaration the kernel already retired — this pick's
    // or any earlier pick's of the goal: it does not re-arm. Only a new
    // declaration can (HZ-0115-3).
    announce_dropped("retired");
    return;
  }
  if (!declaration.region_valid) {
    // Dispatch's own publication, and the state before the producer measures:
    // a declaration with no region exempts nothing. After its handover, the
    // producer dropping the region (its DETACH) ends the pick.
    announce_dropped("no_region");
    if (latched_declaration) {
      retire_grasp_declaration("no_region");
    }
    return;
  }
  const auto& region = declaration.region;
  const auto reject = [&](const char* reason) {
    // A producer error refuses the message's workspace too (#349 review E); a stale
    // measurement is no error, the workspace carries its own confirmation stamp.
    grasp_declaration_refused_ = std::strcmp(reason, "region_stale") != 0;
    if (was_valid) {
      RCLCPP_INFO(this->get_logger(), "safety.grasp_region_dropped reason=%s target=%s", reason,
                  previous_target.c_str());
    }
    if (grasp_region_refusal_reason_ == reason &&
        grasp_region_refusal_target_ == declaration.target_id) {
      return;
    }
    grasp_region_refusal_reason_ = reason;
    grasp_region_refusal_target_ = declaration.target_id;
    RCLCPP_WARN(this->get_logger(),
                "safety.grasp_region_rejected reason=%s target=%s links=%zu half_m=%g,%g,%g "
                "region_frame=%s grid_frame=%s rskill=%s trace=%s evidence=%s",
                reason, declaration.target_id.c_str(), declaration.contact_links.size(),
                region.half_extents.x, region.half_extents.y, region.half_extents.z,
                region.frame_id.c_str(), voxel_frame_id_.c_str(), declaration.rskill_id.c_str(),
                declaration.trace_id.c_str(), region.evidence_ref.c_str());
  };
  // The grasp claim itself was wrong (not a release): its workspace goes with it.
  const auto note_wrong_claim = [&]() {
    grasp_wrong_claim_ = true;
    grasp_wrong_claim_hash_ = std::hash<std::string>{}(declaration.target_id);
    grasp_wrong_claim_stamp_ns_ = declaration.stamp_ns;
  };
  if (declaration.target_id.empty()) {
    reject("no_target");
    return;
  }
  if (!std::isfinite(declaration.timeout_s) || declaration.timeout_s <= 0.0 ||
      declaration.timeout_s > kMaxGraspDeclarationTimeoutS) {
    reject("timeout_out_of_range");
    return;
  }
  if (voxel_frame_id_.empty() || region.frame_id != voxel_frame_id_) {
    reject("frame_mismatch");
    return;
  }
  if (!region.geometry.empty()) {
    reject("geometry_not_supported");  // v1: a grasp region is a box only
    return;
  }
  // Measurement age. Exempt once this declaration has latched at its handover
  // edge: the box is frozen there (producer updates are ignored), the fingers
  // occlude the target so nothing can re-measure it, and the exemption is
  // bounded instead by the carried payload's measured origin staying in the
  // latched box, the stream deadline and timeout_s. Requiring a fresh
  // re-measurement would retire a valid handover with the fingers closed on
  // the target. Up to and including the handover edge the box must be fresh.
  if (!latched_declaration && !region_measurement_fresh(region.stamp_ns, grasp_region_max_age_s_)) {
    reject("region_stale");
    return;
  }
  // The declaration's contact links must be a non-empty subset of the
  // launch-derived allowlist; the exempt mask is exactly that subset. One name
  // off the allowlist refuses the whole declaration rather than trimming it: a
  // producer naming a link the deployment never allowed is not one to trust
  // with the rest (HZ-0115-5).
  if (declaration.contact_links.empty()) {
    reject("no_contact_links");
    return;
  }
  std::bitset<kMaxGraspMaskLinks> mask;
  for (const auto& name : declaration.contact_links) {
    const auto it = std::find(collision_link_names_.begin(), collision_link_names_.end(), name);
    const auto index = static_cast<std::size_t>(it - collision_link_names_.begin());
    if (it == collision_link_names_.end() || index >= kMaxGraspMaskLinks ||
        !grasp_allowlist_[index]) {
      reject("link_not_allowed");
      return;
    }
    mask.set(index);
  }
  // The other jaw of the same gripper: a link posed from the SAME joint column
  // as a declared contact link (the follower finger, a `kMimic*` link of the
  // gripper joint, or the leader when the producer named the follower) is the
  // same hand's finger and is exempt with it — the exemption is per gripper,
  // not per jaw. Only an allowlisted link joins; one off the allowlist stays
  // checked (and fails the declaration if named, above).
  for (std::size_t c = 0; c < kMaxGraspMaskLinks && c < collision_model_.n_links; ++c) {
    const int d = collision_model_.dof_index[c];
    if (!mask[c] || d < 0) {
      continue;
    }
    for (std::size_t k = 0; k < kMaxGraspMaskLinks && k < collision_model_.n_links; ++k) {
      if (collision_model_.dof_index[k] == d && grasp_allowlist_[k]) {
        mask.set(k);
      }
    }
  }
  // One hand per declaration. The producer may infer the target from the
  // robot's own approach instead of a named one, and it arms one hand at a
  // time; the kernel holds it to that rather than trusting it (HZ-0115-12).
  if (!grasp_links_one_hand(collision_model_.parent, grasp_allowlist_, mask)) {
    reject("links_span_hands");
    return;
  }
  const Transform pose = transform_from_translation_quat(
      region.pose.position.x, region.pose.position.y, region.pose.position.z,
      region.pose.orientation.x, region.pose.orientation.y, region.pose.orientation.z,
      region.pose.orientation.w);
  const Vec3 half{region.half_extents.x, region.half_extents.y, region.half_extents.z};
  const GraspRegionStatus status = is_unit_quaternion(region.pose.orientation)
                                       ? ingest_grasp_region(pose, half, mask, grasp_region_)
                                       : GraspRegionStatus::kBadPose;
  if (status != GraspRegionStatus::kOk) {
    reject(grasp_region_status_reason(status));
    return;
  }
  // Handover: the declared object (or, for an empty object_id, the first one
  // carried on the declaring gripper) is attached. From here the per-candidate
  // rule applies — the exemption lives only while that payload's origin stays
  // in the region. Only an attachment on the declaring gripper's own chain can
  // be the handover: its attach link is a declared contact link or an ancestor
  // of one, and never the collision root (a released payload frozen in the
  // base frame, `freeze_released_attachment`, is attached there, and the root
  // is an ancestor of every link). Bimanual: the other hand's payload is on
  // neither chain, so it can neither stand in for the handover nor keep it.
  const std::string* attached_on_gripper = nullptr;
  const char* binding_refusal = nullptr;
  const std::size_t objects = attached_model_.n_objects;
  for (std::size_t i = 0; i < objects && i < attached_labels_.size(); ++i) {
    if (!attachment_on_grasp_chain(attached_model_.objects[i].attach_link, mask)) {
      continue;
    }
    attached_on_gripper = &attached_labels_[i];
    if (declaration.object_id.empty() || attached_labels_[i] == declaration.object_id) {
      // #418 K3/K6: a tracked binding is the handover only while it is at the
      // hand, and never under a (id, epoch) a past handover retired.
      const AttachedBinding& binding = attached_binding_[i];
      if (binding.off_hand != nullptr) {
        binding_refusal = "binding_off_hand";
      } else if (binding.tracked &&
                 tracked_retired_.contains(std::hash<std::string>{}(attached_labels_[i]),
                                           binding.epoch)) {
        binding_refusal = "object_retired";
      } else {
        grasp_region_.handover = true;
        grasp_region_.object_index = i;
      }
      break;
    }
  }
  if (binding_refusal != nullptr) {
    // A claim the kernel cannot verify: refused and retired for good (only a
    // new declaration re-arms), its workspace with it. The payload itself stays
    // a checked payload.
    RCLCPP_WARN(this->get_logger(),
                "safety.grasp_region_rejected reason=%s target=%s object=%s rskill=%s trace=%s",
                binding_refusal, declaration.target_id.c_str(), attached_on_gripper->c_str(),
                declaration.rskill_id.c_str(), declaration.trace_id.c_str());
    note_wrong_claim();
    retire_grasp_declaration(binding_refusal);
    return;
  }
  if (latched_declaration && !grasp_region_.handover) {
    // This declaration's handover latched, and nothing is attached on the
    // declaring chain any more: the gripper released its payload. The pick is
    // over even when the attachment set is not empty — the other hand still
    // holds, or the released payload is frozen on the base through its release
    // window — so the declaration retires now (HZ-0115-3), not at timeout_s.
    retire_grasp_declaration("released");
    return;
  }
  if (attached_on_gripper != nullptr && !grasp_region_.handover) {
    // The declaring gripper attached something the declaration does not name.
    // Whatever it grasped, it is not the scene the region was measured for, and
    // without a handover the exemption would live on to timeout_s. Fail closed:
    // retire it for good (HZ-0115-3: only a new declaration re-arms). A
    // producer contradicting the attachment stream is a rejection, not a
    // routine disarm, so it is WARNed with what was declared vs attached. Once
    // per declaration: its later heartbeats take the `retired` path above.
    RCLCPP_WARN(this->get_logger(),
                "safety.grasp_region_rejected reason=handover_object_mismatch target=%s "
                "object=%s attached=%s rskill=%s trace=%s",
                declaration.target_id.c_str(), declaration.object_id.c_str(),
                attached_on_gripper->c_str(), declaration.rskill_id.c_str(),
                declaration.trace_id.c_str());
    note_wrong_claim();
    retire_grasp_declaration("handover_object_mismatch");
    return;
  }
  // Latch at the handover edge: the first snapshot that sees the declared object
  // attached freezes the box for this (target, stamp). Later snapshots of the
  // same declaration keep the latched box whatever region they carry — the
  // producer cannot extend the exemption by moving the box with the payload.
  // Before any handover the region still updates on every snapshot.
  if (grasp_latched_ && grasp_latched_target_ == declaration.target_id &&
      grasp_latched_stamp_ns_ == declaration.stamp_ns) {
    // The latch froze who the box exempts as well as where it is: the same
    // (target, stamp) naming the other hand, or another object that happens to
    // sit in the frozen box, is not the pick that latched. Fail closed.
    const std::string& held = attached_labels_[grasp_region_.object_index];
    if (grasp_region_.link_mask != grasp_latched_region_.link_mask ||
        held != grasp_latched_object_ ||
        attached_binding_[grasp_region_.object_index].epoch != grasp_latched_epoch_) {
      RCLCPP_WARN(this->get_logger(),
                  "safety.grasp_region_rejected reason=latch_mismatch target=%s latched_object=%s "
                  "object=%s latched_links=%zu links=%zu rskill=%s trace=%s",
                  declaration.target_id.c_str(), grasp_latched_object_.c_str(), held.c_str(),
                  grasp_latched_region_.link_mask.count(), grasp_region_.link_mask.count(),
                  declaration.rskill_id.c_str(), declaration.trace_id.c_str());
      note_wrong_claim();
      retire_grasp_declaration("latch_mismatch");
      return;
    }
    double moved = 0.0;
    for (std::size_t k = 0; k < 9; ++k) {
      moved = std::max(moved, std::abs(grasp_region_.pose.r[k] - grasp_latched_region_.pose.r[k]));
    }
    const Vec3& t = grasp_region_.pose.t;
    const Vec3& lt = grasp_latched_region_.pose.t;
    const Vec3& h = grasp_region_.half_extents;
    const Vec3& lh = grasp_latched_region_.half_extents;
    moved = std::max({moved, std::abs(t.x - lt.x), std::abs(t.y - lt.y), std::abs(t.z - lt.z),
                      std::abs(h.x - lh.x), std::abs(h.y - lh.y), std::abs(h.z - lh.z)});
    if (moved > kGraspRegionLatchNoise && !grasp_latched_moved_warned_) {
      grasp_latched_moved_warned_ = true;
      RCLCPP_WARN(this->get_logger(),
                  "safety.grasp_region_moved_after_handover target=%s max_delta=%g ignored",
                  declaration.target_id.c_str(), moved);
    }
    grasp_region_.pose = grasp_latched_region_.pose;
    grasp_region_.half_extents = grasp_latched_region_.half_extents;
  } else if (grasp_region_.handover) {
    grasp_latched_ = true;
    grasp_latched_target_ = declaration.target_id;
    grasp_latched_stamp_ns_ = declaration.stamp_ns;
    grasp_latched_region_ = grasp_region_;
    grasp_latched_object_ = attached_labels_[grasp_region_.object_index];
    grasp_latched_epoch_ = attached_binding_[grasp_region_.object_index].epoch;
    grasp_latched_tracked_ = attached_binding_[grasp_region_.object_index].tracked;
    grasp_latched_moved_warned_ = false;
    RCLCPP_INFO(this->get_logger(), "safety.grasp_region_latched target=%s at handover",
                declaration.target_id.c_str());
  }
  grasp_region_refusal_reason_.clear();
  grasp_region_refusal_target_.clear();
  grasp_region_stamp_ns_ = region.stamp_ns;
  grasp_armed_target_ = declaration.target_id;
  grasp_armed_stamp_ns_ = declaration.stamp_ns;
  if (!was_valid || previous_target != declaration.target_id) {
    RCLCPP_INFO(this->get_logger(),
                "safety.grasp_region_armed target=%s links=%zu half_m=%g,%g,%g rskill=%s trace=%s "
                "evidence=%s",
                declaration.target_id.c_str(), mask.count(), half.x, half.y, half.z,
                declaration.rskill_id.c_str(), declaration.trace_id.c_str(),
                region.evidence_ref.c_str());
  }
  if (grasp_region_.handover && !was_handover) {
    RCLCPP_INFO(this->get_logger(), "safety.grasp_region_handover target=%s object=%s",
                declaration.target_id.c_str(),
                attached_labels_[grasp_region_.object_index].c_str());
  }
}

void SafetyKernelLifecycleNode::ingest_tracked_declaration_key(
    const openral_msgs::msg::WorldStateStamped& msg) {
  tracked_decl_active_ = false;
  if (!msg.grasp_declaration_valid) {
    return;
  }
  const auto& d = msg.grasp_declaration;
  // The same identity rules as the region path: active, a target, a bounded
  // backstop, and never a declaration the kernel retired (HZ-0115-3).
  if (!d.active || d.target_id.empty() || !std::isfinite(d.timeout_s) || d.timeout_s <= 0.0 ||
      d.timeout_s > kMaxGraspDeclarationTimeoutS ||
      grasp_retired_.contains(std::hash<std::string>{}(d.target_id), d.stamp_ns)) {
    return;
  }
  // The tracker's `declaration_key(target_id, stamp_ns)`: "<target_id>@<stamp_ns>".
  tracked_decl_key_ = d.target_id + "@" + std::to_string(d.stamp_ns);
  tracked_decl_key_hash_ = std::hash<std::string>{}(tracked_decl_key_);
  tracked_decl_stamp_ns_ = d.stamp_ns;
  tracked_decl_timeout_s_ = d.timeout_s;
  tracked_decl_active_ = true;
}

bool SafetyKernelLifecycleNode::refresh_tracked_cover() {
  const bool frame_ok = !voxel_frame_id_.empty() && tracked_set_frame_ == voxel_frame_id_;
  if (tracked_cover_dirty_) {
    // Once per grid (or set): the kernel's own cover check, on its own grid.
    const std::int64_t now = this->now().nanoseconds();
    const auto max_age_ns = static_cast<std::int64_t>(grasp_region_max_age_s_ * 1e9);
    bool fk_done = false;
    std::array<CoverPass, kMaxTrackedObjects> next{};
    std::size_t next_n = 0;
    for (std::size_t k = 0; k < tracked_n_; ++k) {
      TrackedEntry& e = tracked_[k];
      const bool usable = frame_ok && !voxel_overflow_;
      e.cover_ok = usable && tracked_cover_sufficient(
                                 tracked_box_cover(voxel_grid_, e.box.pose, e.box.half_extents),
                                 e.visible_cells);
      const CoverPass* prev = nullptr;
      for (std::size_t p = 0; p < cover_pass_n_; ++p) {
        if (cover_pass_[p].id_hash == e.id_hash && cover_pass_[p].epoch == e.epoch) {
          prev = &cover_pass_[p];
        }
      }
      CoverPass rec{e.id_hash, e.epoch, e.cover_ok ? now : 0, false};
      if (!e.cover_ok && usable && e.contact && prev != nullptr && prev->pass_ns > 0 &&
          now - prev->pass_ns >= 0 && now - prev->pass_ns <= max_age_ns &&
          workspace_contact_enabled_ && workspace_region_.valid && measured_state_fresh()) {
        // Contact hold: a hand at the target hides or bisects it (Isaac R41-R43: 39 of 79
        // expected cells), so the shortfall is not the object gone; the destination it
        // places into likewise (R63: the bowl). Never re-stamped: the hold ends
        // `grasp_region_max_age_s` after the kernel's own last pass.
        if (!fk_done) {
          std::copy(q_meas_.begin(), q_meas_.end(), q_fk_.begin());
          for (const int d : collision_base_dofs_) {
            q_fk_[static_cast<std::size_t>(d)] = 0.0;
          }
          forward_kinematics(collision_model_, q_fk_.data(), q_fk_.size(), binding_scratch_);
          fk_done = true;
        }
        if (box_meets_hand_links(collision_model_, binding_scratch_, workspace_region_.link_mask,
                                 e.box.pose, e.box.half_extents, voxel_grid_.resolution)) {
          e.cover_ok = true;
          rec.pass_ns = prev->pass_ns;
          rec.held = true;
          if (!prev->held) {
            RCLCPP_WARN(this->get_logger(),
                        "safety.tracked_cover_held object=%s epoch=%u last_pass_age_s=%.3f "
                        "max_age_s=%.3f (issue #418 contact hold, Safety-WG review pending)",
                        e.object_id.c_str(), e.epoch,
                        static_cast<double>(now - prev->pass_ns) * 1e-9, grasp_region_max_age_s_);
          }
        }
      }
      if (rec.pass_ns > 0 && next_n < next.size()) {
        next[next_n++] = rec;
      }
    }
    cover_pass_ = next;
    cover_pass_n_ = next_n;
    tracked_cover_dirty_ = false;
  }
  return frame_ok;
}

bool SafetyKernelLifecycleNode::tracked_stream_fresh() {
  const double age_s = (this->now() - attached_stamp_).seconds();
  return attached_received_ && !attached_overflow_ && age_s >= 0.0 &&
         age_s <= attached_collision_deadline_s_;
}

std::size_t SafetyKernelLifecycleNode::fill_tracked_grasp() {
  tracked_grasp_.n_objects = 0;
  const bool frame_ok = refresh_tracked_cover();
  // The declaration's own liveness: its backstop, no future stamp, and the
  // stream carrying it fresh (exactly `grasp_declaration_live`'s rules).
  const std::int64_t now_ns = this->now().nanoseconds();
  const std::int64_t elapsed_ns = now_ns - tracked_decl_stamp_ns_;
  const bool declaration_live =
      tracked_decl_active_ && elapsed_ns >= 0 &&
      elapsed_ns <= static_cast<std::int64_t>(tracked_decl_timeout_s_ * 1e9) &&
      tracked_stream_fresh();
  if (declaration_live && frame_ok) {
    for (std::size_t k = 0; k < tracked_n_; ++k) {
      const TrackedEntry& e = tracked_[k];
      if (!e.target || !e.exemptible || !e.cover_ok || e.declaration_key != tracked_decl_key_ ||
          tracked_retired_.contains(e.id_hash, e.epoch) ||
          !region_measurement_fresh(e.confirmed_ns, grasp_region_max_age_s_)) {
        continue;
      }
      const std::size_t n = tracked_grasp_.n_objects++;
      tracked_grasp_.box[n] = e.box.pose;
      tracked_grasp_.half_extents[n] = e.box.half_extents;
    }
  }
  if (tracked_grasp_.n_objects != tracked_grasp_logged_n_) {
    tracked_grasp_logged_n_ = tracked_grasp_.n_objects;
    RCLCPP_INFO(this->get_logger(),
                "safety.tracked_grasp_eligible n=%zu of=%zu declaration=%s live=%d",
                tracked_grasp_.n_objects, tracked_n_, tracked_decl_key_.c_str(),
                static_cast<int>(declaration_live));
  }
  return tracked_grasp_.n_objects;
}

namespace {

// Everything `mark_support_cells` reads besides the grid itself, as one bit-comparable
// array: the explaining objects' boxes and supports, and the workspace's pose / extent.
template <std::size_t N>
void support_key(const WorkspaceExplanation& e, const WorkspaceContactRegion& ws,
                 std::array<double, N>& key) noexcept {
  static_assert(N == 1 + kMaxTrackedObjects * 16 + 16, "one slot per input");
  std::size_t k = 0;
  const std::size_t n = std::min(e.n_objects, kMaxTrackedObjects);
  key[k++] = static_cast<double>(n);
  for (std::size_t i = 0; i < n; ++i) {
    key[k++] = e.box[i].t.x;
    key[k++] = e.box[i].t.y;
    key[k++] = e.box[i].t.z;
    for (int r = 0; r < 9; ++r) {
      key[k++] = e.box[i].r[r];
    }
    key[k++] = e.half_extents[i].x;
    key[k++] = e.half_extents[i].y;
    key[k++] = e.half_extents[i].z;
    key[k++] = e.support_z[i];
  }
  k = 1 + kMaxTrackedObjects * 16;
  key[k++] = ws.valid ? 1.0 : 0.0;
  key[k++] = ws.pose.t.x;
  key[k++] = ws.pose.t.y;
  key[k++] = ws.pose.t.z;
  for (int r = 0; r < 9; ++r) {
    key[k++] = ws.pose.r[r];
  }
  key[k++] = ws.half_extents.x;
  key[k++] = ws.half_extents.y;
  key[k++] = ws.half_extents.z;
}

}  // namespace

std::size_t SafetyKernelLifecycleNode::fill_workspace_explained() {
  WorkspaceExplanation& e = workspace_explained_;
  e.n_objects = 0;
  const bool frame_ok = refresh_tracked_cover();
  // Any role, any declaration: a neighbour carton or the destination plate
  // explains its own cells. Never split (a foreign body in the box),
  // tentative, released or lost; never a retired identity, a stale
  // confirmation, a box the kernel's own grid does not cover, or a set the
  // stream no longer carries fresh. A BOUND one explains only its own box's
  // cells, never a support: the cells the octomap bridge has yet to clear as
  // the payload's (Spark R245/R253, 2026-10-22: the bind dropped the banana from
  // this set and the fingertip in its still-mapped cells stopped 0.1 s later).
  // The cover check ends it once the bridge clears them. Safety-WG review owed.
  if (frame_ok && tracked_stream_fresh()) {
    for (std::size_t k = 0; k < tracked_n_; ++k) {
      const TrackedEntry& t = tracked_[k];
      const bool bound = t.live && !t.exemptible;
      if ((!t.exemptible && !bound) || !t.cover_ok ||
          tracked_retired_.contains(t.id_hash, t.epoch) ||
          !region_measurement_fresh(t.confirmed_ns, grasp_region_max_age_s_)) {
        continue;
      }
      const std::size_t n = e.n_objects++;
      e.box[n] = t.box.pose;
      e.half_extents[n] = t.box.half_extents;
      e.support_z[n] = bound ? std::numeric_limits<double>::quiet_NaN() : t.support_z;
    }
  }
  const bool static_ok = static_cells_marked_ > 0 && !voxel_overflow_;
  e.static_cells = static_ok ? static_cells_.data() : nullptr;
  e.static_cells_n = static_ok ? static_cast<std::size_t>(voxel_grid_.sx) *
                                     static_cast<std::size_t>(voxel_grid_.sy) *
                                     static_cast<std::size_t>(voxel_grid_.sz)
                               : 0;
  // The connected support layer of these objects on this grid, re-marked only
  // when one of its inputs changed (it is a whole-grid pass, not a per-check one).
  e.support_cells = nullptr;
  e.support_cells_n = 0;
  if (voxel_overflow_ || support_cells_.empty()) {
    support_cells_marked_ = 0;
  } else {
    std::array<double, kSupportKeyN> key{};
    support_key(e, voxel_grid_.workspace_region, key);
    if (support_cells_dirty_ ||
        std::memcmp(key.data(), support_key_.data(), sizeof(double) * kSupportKeyN) != 0) {
      support_key_ = key;
      support_cells_dirty_ = false;
      support_cells_marked_ =
          mark_support_cells(voxel_grid_, e, support_cells_.data(), support_cells_.size(),
                             support_queue_.data(), support_queue_.size());
    }
    if (support_cells_marked_ > 0) {
      e.support_cells = support_cells_.data();
      e.support_cells_n = static_cast<std::size_t>(voxel_grid_.sx) *
                          static_cast<std::size_t>(voxel_grid_.sy) *
                          static_cast<std::size_t>(voxel_grid_.sz);
    }
  }
  if (e.n_objects != workspace_explained_logged_n_) {
    workspace_explained_logged_n_ = e.n_objects;
    RCLCPP_INFO(this->get_logger(),
                "safety.workspace_explained objects=%zu of=%zu static_cells=%zu "
                "support_cells=%zu",
                e.n_objects, tracked_n_, static_ok ? static_cells_marked_ : std::size_t{0},
                support_cells_marked_);
  }
  return e.n_objects;
}

void SafetyKernelLifecycleNode::refresh_static_cells() {
  if (static_points_.empty()) {
    return;
  }
  const VoxelGrid& g = voxel_grid_;
  const VoxelGrid& c = static_cells_for_;
  bool same = static_cells_for_valid_ && g.sx == c.sx && g.sy == c.sy && g.sz == c.sz &&
              g.resolution == c.resolution && g.pose.t.x == c.pose.t.x &&
              g.pose.t.y == c.pose.t.y && g.pose.t.z == c.pose.t.z;
  for (int i = 0; same && i < 9; ++i) {
    same = g.pose.r[i] == c.pose.r[i];
  }
  if (same) {
    return;
  }
  static_cells_for_ = voxel_grid_;
  static_cells_for_.occupancy = nullptr;
  static_cells_for_valid_ = true;
  // The file names cells of the base frame's `resolution` lattice (the launch
  // writes one centre per cell the bridge marks; a fixed-base octree accumulates
  // in the base frame). On any other lattice a centre could land in a
  // neighbouring cell, which would explain a cell that is not static: none then.
  const auto on_lattice = [&](double v) {
    const double k = v / g.resolution;
    return std::isfinite(k) && std::fabs(k - std::round(k)) <= 1e-6;
  };
  bool aligned = on_lattice(g.pose.t.x) && on_lattice(g.pose.t.y) && on_lattice(g.pose.t.z);
  for (int i = 0; aligned && i < 9; ++i) {
    aligned = std::fabs(g.pose.r[i] - ((i % 4 == 0) ? 1.0 : 0.0)) <= 1e-9;
  }
  static_cells_lattice_ok_ = aligned;
  if (!aligned) {
    static_cells_marked_ = 0;
    RCLCPP_WARN(this->get_logger(),
                "safety.workspace_static_cells refused reason=lattice: the grid is not the base "
                "frame's %g m lattice, so no cell is explained as declared static geometry",
                g.resolution);
    return;
  }
  static_cells_marked_ = mark_static_cells(voxel_grid_, static_points_, static_cells_.data(),
                                           static_cells_.size(), kExplainedStaticPadCells);
  RCLCPP_INFO(this->get_logger(),
              "safety.workspace_static_cells marked=%zu of %zu points (+%zu cell pad)",
              static_cells_marked_, static_points_.size(), kExplainedStaticPadCells);
}

bool SafetyKernelLifecycleNode::load_static_points(const std::string& path, std::vector<Vec3>& out,
                                                   std::string& error) {
  // The octomap bridge's `load_points_file` format: one `x y z` per line, blank
  // lines and `#` comments skipped; anything else refuses the whole file.
  out.clear();
  std::ifstream in(path);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') {
      continue;
    }
    std::istringstream fields(line);
    Vec3 p{};
    if (!(fields >> p.x >> p.y >> p.z) || !std::isfinite(p.x) || !std::isfinite(p.y) ||
        !std::isfinite(p.z)) {
      error = path + ":" + std::to_string(lineno) + ": expected three finite numbers";
      out.clear();
      return false;
    }
    out.push_back(p);
  }
  if (out.empty()) {
    error = path + ": no points";
    return false;
  }
  return true;
}

bool SafetyKernelLifecycleNode::attachment_on_grasp_chain(
    int attach_link, const std::bitset<kMaxGraspMaskLinks>& mask) const noexcept {
  const auto& parent = collision_model_.parent;
  if (attach_link < 0 || static_cast<std::size_t>(attach_link) >= parent.size()) {
    return false;
  }
  const auto a = static_cast<std::size_t>(attach_link);
  if (a < kMaxGraspMaskLinks && mask[a]) {
    return true;  // attached to a declared contact link itself
  }
  if (parent[a] < 0) {
    return false;  // the root: every link's ancestor, and where released payloads are frozen
  }
  // Is `attach_link` an ancestor of a declared contact link? Parents index
  // below their children, so each walk ends at the root in < n_links steps.
  for (std::size_t c = 0; c < kMaxGraspMaskLinks && c < parent.size(); ++c) {
    if (!mask[c]) {
      continue;
    }
    for (int l = parent[c]; l >= 0; l = parent[static_cast<std::size_t>(l)]) {
      if (l == attach_link) {
        return true;
      }
    }
  }
  return false;
}

bool SafetyKernelLifecycleNode::grasp_declaration_live() const noexcept {
  if (!grasp_region_.valid || grasp_declaration_timeout_s_ <= 0.0) {
    return false;
  }
  const std::int64_t now_ns = this->now().nanoseconds();
  // Backstop and future stamps, exactly as for the place declaration.
  const std::int64_t elapsed_ns = now_ns - grasp_declaration_stamp_ns_;
  if (elapsed_ns < 0 ||
      elapsed_ns > static_cast<std::int64_t>(grasp_declaration_timeout_s_ * 1e9)) {
    return false;
  }
  // The stream that carries the region must itself be fresh. Stale means no
  // exemption — never a drop by itself; with attached checking on, the
  // attachment gate has already refused the chunk before this is read.
  const double age_s = (this->now() - attached_stamp_).seconds();
  if (!(attached_received_ && !attached_overflow_ && age_s >= 0.0 &&
        age_s <= attached_collision_deadline_s_)) {
    return false;
  }
  // The region's own measurement, ageing between messages; not applied once
  // the handover has latched the box (see ingest_grasp_declaration).
  return grasp_region_.handover ||
         region_measurement_fresh(grasp_region_stamp_ns_, grasp_region_max_age_s_);
}

void SafetyKernelLifecycleNode::drop_workspace(const char* reason) {
  if (workspace_region_.valid) {
    RCLCPP_INFO(this->get_logger(), "safety.workspace_region_dropped reason=%s target=%s", reason,
                workspace_target_.c_str());
    RCLCPP_INFO(this->get_logger(),
                "safety.workspace_contact_summary target=%s pairs=%llu max_depth_mm=%.1f "
                "max_speed_mps=%.3f max_speed_source=%s max_mps=%g max_lead_m=%.3f "
                "lead_bound_m=%g",
                workspace_target_.c_str(), static_cast<unsigned long long>(workspace_pairs_total_),
                workspace_pairs_total_ > 0 ? std::max(0.0, -workspace_min_distance_total_) * 1e3
                                           : 0.0,
                workspace_speed_max_seen_, workspace_speed_max_measured_ ? "measured" : "commanded",
                workspace_max_speed_mps_, workspace_lead_max_seen_, workspace_max_lead_m_);
    clear_workspace_commands(ResetReason::kDeclarationChanged);  // start the chain afresh
  }
  workspace_region_ = WorkspaceContactRegion{};
  voxel_grid_.workspace_region = WorkspaceContactRegion{};
  workspace_pairs_total_ = 0;
  workspace_min_distance_total_ = std::numeric_limits<double>::infinity();
  workspace_speed_max_seen_ = 0.0;
  workspace_speed_max_measured_ = false;
  workspace_lead_max_seen_ = 0.0;
}

void SafetyKernelLifecycleNode::ingest_workspace_contact(
    const openral_msgs::msg::WorldStateStamped& msg) {
  const std::string previous_target = workspace_target_;
  const bool was_valid = workspace_region_.valid;
  // A refusal the producer keeps re-sending is a standing state: WARN once per
  // change of (reason, target), drop silently after.
  const auto reject = [&](const char* reason, const std::string& target) {
    drop_workspace(reason);
    const std::string key = std::string(reason) + ":" + target;
    if (workspace_refusal_reason_ == key) {
      return;
    }
    workspace_refusal_reason_ = key;
    RCLCPP_WARN(this->get_logger(), "safety.workspace_region_rejected reason=%s target=%s", reason,
                target.c_str());
  };
  if (!msg.grasp_declaration_valid) {
    drop_workspace("no_declaration");
    return;
  }
  const auto& declaration = msg.grasp_declaration;
  if (!declaration.active) {
    drop_workspace("retracted");
    return;
  }
  if (!declaration.workspace_region_valid) {
    drop_workspace("no_workspace");
    workspace_refusal_reason_.clear();
    return;
  }
  if (grasp_declaration_refused_) {
    // The kernel refused this declaration's grasp part (e.g. a link off the
    // allowlist): the same producer's workspace is not trusted either (#349 review E).
    reject("grasp_refused", declaration.target_id);
    return;
  }
  if (grasp_wrong_claim_ &&
      grasp_wrong_claim_hash_ == std::hash<std::string>{}(declaration.target_id) &&
      grasp_wrong_claim_stamp_ns_ == declaration.stamp_ns) {
    // Retired because its grasp claim was wrong (the gripper holds another object, or
    // the latched pick changed): the workspace it armed goes too, for good. A correct
    // handover's retirement keeps it (p12_mix04, above).
    reject("grasp_claim_wrong", declaration.target_id);
    return;
  }
  if (declaration.target_id.empty()) {
    reject("no_target", declaration.target_id);
    return;
  }
  if (!std::isfinite(declaration.timeout_s) || declaration.timeout_s <= 0.0 ||
      declaration.timeout_s > kMaxGraspDeclarationTimeoutS) {
    reject("timeout_out_of_range", declaration.target_id);
    return;
  }
  const std::int64_t elapsed_ns = this->now().nanoseconds() - declaration.stamp_ns;
  if (elapsed_ns < 0 || elapsed_ns > static_cast<std::int64_t>(declaration.timeout_s * 1e9)) {
    drop_workspace("expired");
    return;
  }
  // A retired identity (handover exit, a release) ends its GRASP REGION, which
  // belonged to that pick — not the workspace (a REFUSED grasp part does, above): carry, place and
  // a regrasp happen in the same declared, map-confirmed workspace (#349 p12_mix04 / p12_mix07
  // stopped right after a handover_exit / detach dropped it). The workspace ends
  // with the declaration itself: retraction, expiry, a stale confirmation.
  const std::size_t target_hash = std::hash<std::string>{}(declaration.target_id);
  const auto& region = declaration.workspace_region;
  if (voxel_frame_id_.empty() || region.frame_id != voxel_frame_id_) {
    reject("frame_mismatch", declaration.target_id);
    return;
  }
  if (!region.geometry.empty()) {
    reject("geometry_not_supported", declaration.target_id);
    return;
  }
  if (!region_measurement_fresh(region.stamp_ns, grasp_region_max_age_s_)) {
    reject("region_stale", declaration.target_id);
    return;
  }
  const Transform pose = transform_from_translation_quat(
      region.pose.position.x, region.pose.position.y, region.pose.position.z,
      region.pose.orientation.x, region.pose.orientation.y, region.pose.orientation.z,
      region.pose.orientation.w);
  const Vec3 half{region.half_extents.x, region.half_extents.y, region.half_extents.z};
  WorkspaceContactRegion candidate;
  const GraspRegionStatus status =
      is_unit_quaternion(region.pose.orientation)
          ? ingest_workspace_region(
                pose, half, workspace_contact_enabled_ ? workspace_links_ : provider_link_bits_,
                workspace_max_speed_mps_, workspace_max_lead_m_, candidate)
          : GraspRegionStatus::kBadPose;
  if (status != GraspRegionStatus::kOk) {
    reject(grasp_region_status_reason(status), declaration.target_id);
    return;
  }
  if (was_valid && previous_target != declaration.target_id) {
    drop_workspace("target_changed");  // the summary belongs to the previous identity
  }
  const bool announce = !workspace_region_.valid;
  workspace_region_ = candidate;
  workspace_region_stamp_ns_ = region.stamp_ns;
  workspace_declaration_stamp_ns_ = declaration.stamp_ns;
  workspace_declaration_timeout_s_ = declaration.timeout_s;
  workspace_target_ = declaration.target_id;
  workspace_target_hash_ = target_hash;
  workspace_refusal_reason_.clear();
  if (announce) {
    clear_workspace_commands(ResetReason::kDeclarationChanged);  // start the chain afresh
    RCLCPP_INFO(this->get_logger(),
                "safety.workspace_region_armed target=%s links=%zu half_m=%g,%g,%g "
                "max_speed_mps=%g max_lead_m=%g rskill=%s trace=%s evidence=%s",
                declaration.target_id.c_str(), workspace_links_.count(), half.x, half.y, half.z,
                workspace_max_speed_mps_, workspace_max_lead_m_, declaration.rskill_id.c_str(),
                declaration.trace_id.c_str(), region.evidence_ref.c_str());
  }
}

bool SafetyKernelLifecycleNode::workspace_live() const noexcept {
  if (!workspace_ingest_enabled_ || !workspace_region_.valid ||
      workspace_declaration_timeout_s_ <= 0.0) {
    return false;
  }
  const std::int64_t elapsed_ns = this->now().nanoseconds() - workspace_declaration_stamp_ns_;
  if (elapsed_ns < 0 ||
      elapsed_ns > static_cast<std::int64_t>(workspace_declaration_timeout_s_ * 1e9)) {
    return false;
  }
  // A retirement (e.g. `handover_exit`) ends the grasp region, not the workspace
  // (see `ingest_workspace_contact`).
  const double age_s = (this->now() - attached_stamp_).seconds();
  if (!(attached_received_ && !attached_overflow_ && age_s >= 0.0 &&
        age_s <= attached_collision_deadline_s_)) {
    return false;
  }
  return region_measurement_fresh(workspace_region_stamp_ns_, grasp_region_max_age_s_);
}

void SafetyKernelLifecycleNode::workspace_snapshot(std::vector<Transform>& out) const noexcept {
  for (const int l : workspace_link_index_) {
    out[static_cast<std::size_t>(l)] = collision_scratch_.link_world[static_cast<std::size_t>(l)];
  }
}

double SafetyKernelLifecycleNode::workspace_hand_speed(std::vector<Transform>& prev) noexcept {
  double moved = 0.0;
  for (const int l : workspace_link_index_) {
    const auto i = static_cast<std::size_t>(l);
    const Transform& now = collision_scratch_.link_world[i];
    moved = std::max(moved, link_displacement_bound(prev[i], now, workspace_reach(i)));
    prev[i] = now;
  }
  const double speed = moved / workspace_control_period_s_;
  return std::isfinite(speed) ? speed : std::numeric_limits<double>::infinity();
}

double SafetyKernelLifecycleNode::workspace_reach(std::size_t link) const noexcept {
  double reach = workspace_link_reach_[link];
  if (attached_collision_enabled_) {
    for (std::size_t i = 0; i < attached_model_.n_objects; ++i) {
      const AttachedObject& obj = attached_model_.objects[i];
      if (obj.attach_link == static_cast<int>(link)) {
        reach = std::max(reach, attached_object_reach(attached_model_, obj));
      }
    }
  }
  return reach;
}

double SafetyKernelLifecycleNode::workspace_hand_lead() const noexcept {
  double lead = 0.0;
  for (const int l : workspace_link_index_) {
    const auto i = static_cast<std::size_t>(l);
    lead = std::max(lead, link_displacement_bound(workspace_meas_pose_[i],
                                                  collision_scratch_.link_world[i],
                                                  workspace_link_reach_[i]));
  }
  return std::isfinite(lead) ? lead : std::numeric_limits<double>::infinity();
}

void SafetyKernelLifecycleNode::clear_workspace_commands(ResetReason why) noexcept {
  std::fill(workspace_cmd_set_.begin(), workspace_cmd_set_.end(), std::uint8_t{0});
  workspace_cmd_drops_seen_ = chunks_dropped_;
  if (provider_ != nullptr) {
    provider_->on_reset(why);  // ADR-0121: the provider forgets its chain on the same events
  }
}

bool SafetyKernelLifecycleNode::workspace_command_fresh(std::size_t d,
                                                        std::int64_t now_ns) const noexcept {
  if (d >= workspace_cmd_set_.size() || workspace_cmd_set_[d] == 0) {
    return false;
  }
  const double window_ns = kWorkspaceCommandFreshPeriods * workspace_control_period_s_ * 1e9;
  const std::int64_t since_due = now_ns - workspace_cmd_due_ns_[d];
  // Not yet due: the robot is still on an earlier row of that chunk, which the
  // kernel cannot name — fall back to measured.
  return since_due >= 0 && static_cast<double>(since_due) <= window_ns;
}

}  // namespace openral_safety_kernel

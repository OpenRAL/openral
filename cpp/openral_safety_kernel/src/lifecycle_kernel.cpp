// SPDX-License-Identifier: Apache-2.0
// SafetyKernelLifecycleNode source.

#include "openral_safety_kernel/lifecycle_kernel.hpp"

#include "openral_safety_kernel/otel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>

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

rclcpp::QoS chunk_qos() {
  // The openral slot dispatcher publishes N chunks per
  // policy tick on /openral/candidate_action (arm CARTESIAN_DELTA +
  // gripper GRIPPER_POSITION + optional base BODY_TWIST). KEEP_LAST=1
  // on the subscriber side coalesces back-to-back publishes inside
  // the same callback batch: only the last slot's chunk survives, so
  // in deploy_sim the arm freezes while the gripper keeps streaming
  // because that's the LAST published slot per tick. Four-slot mobile
  // manipulation plus predictive collision work can backlog beyond ten
  // samples; fifty holds twelve complete ticks while staying bounded.
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

/// Decode one wire `AttachedCollisionPrimitive` into the kernel's input record.
///
/// Shared by the attached-payload ingest and the declared place target's
/// geometry (ADR-0098): both read the same message type off the same
/// `WorldStateStamped`, and a shape the two decoded differently would be a
/// payload and a receptacle that disagree about where a surface is. Returns
/// false on an unknown `SHAPE_*` tag or too few dimensions for the tag given —
/// every caller fails closed on that, because an unrecognised shape is not safe
/// to treat as absent.
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
  // Realtime hints — best-effort; we log on failure rather than aborting.
  this->declare_parameter<bool>("request_sched_fifo", false);
  this->declare_parameter<std::vector<std::int64_t>>("cpu_affinity", std::vector<std::int64_t>{});

  // Parameter-based envelope source (2026-05-24). The
  // Python `sim_e2e.launch.py` unpacks `robots/<id>/robot.yaml` via
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
  this->declare_parameter<double>("max_force_n", kPosInfinity);
  this->declare_parameter<double>("max_torque_nm", kPosInfinity);
  this->declare_parameter<double>("contact_force_threshold_n", kPosInfinity);
  this->declare_parameter<bool>("deadman_required", false);

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
  this->declare_parameter<std::vector<std::int64_t>>("collision_allowed_pairs",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<std::string>>("collision_link_names",
                                                    std::vector<std::string>{});

  // World phase — world-obstacle collision check (opt-in). Obstacles
  // arrive on /openral/world_collision in the robot base frame.
  this->declare_parameter<bool>("world_collision_enabled", false);
  this->declare_parameter<double>("world_collision_margin_m", 0.0);
  this->declare_parameter<double>("world_collision_deadline_ms", 500.0);
  this->declare_parameter<std::int64_t>("world_collision_max_primitives", 64);

  // Voxel phase — dense occupancy-grid world check (octomap path).
  this->declare_parameter<bool>("world_voxel_enabled", false);
  this->declare_parameter<double>("world_voxel_margin_m", 0.0);
  this->declare_parameter<double>("world_voxel_deadline_ms", 500.0);
  // Sized by the ROBOT, not by taste: the grid must cover wherever the
  // kernel-checked geometry can reach, and on a lattice-aligned grid that
  // volume is a ball (the grid's axes are the map's and turn relative to the
  // base, so a base-aligned box is not invariant to them). panda_mobile's
  // checked arm reaches 1016 mm from the grid centre; the deploy launch covers
  // 1.05 m, which at the sim's 25 mm cells is 85^3 = 614125 including the one
  // cell per axis the lattice snap can add.
  //
  // The previous 262144 was the cap the sim BOX WAS SHRUNK TO FIT (1.6 m at
  // 25 mm = 64^3) — the inversion this replaces. It left the arm reaching up to
  // 124 mm outside the published grid, where the world check sees nothing.
  //
  // Cost is dominated by the attached-contact baseline, not the occupancy:
  // `attached_contact_distance_` is attached_max_objects x 8 B per cell — 64
  // B/cell against occupancy's 1 B — so this default reserves ~40 MB, of which
  // 39 MB is that baseline. Making it sparse is the lever if that ever binds.
  this->declare_parameter<std::int64_t>("world_voxel_max_cells", 614125);

  // Attached-payload phase — grasped objects carried on
  // /openral/world_state_fast (ADR-0092). Each object leaves world occupancy
  // (openral_octomap_bridge clears the payload's own cells out of
  // /openral/world_voxels off this same message — the kernel exempts nothing on
  // its behalf) and is re-checked as collision-active robot geometry (vs world,
  // voxels, and the robot's own links except its attach link + touch links). Caps
  // are fixed-capacity: an over-capacity, unknown-link, or malformed attachment
  // set fails closed (the next candidate action is dropped until a clean message
  // lands). Each object owns a single identity, attach link, and touch-link set
  // but may carry several primitives; the object, total-primitive, and
  // total-touch-link caps are enforced independently and fail closed.
  this->declare_parameter<bool>("attached_collision_enabled", false);
  this->declare_parameter<double>("attached_collision_margin_m", 0.0);
  // How many consecutive advisory refusals (#176) the kernel will issue before
  // treating the next one as an ordinary latched stop. 0 disables the band
  // entirely and restores the pre-#176 behaviour exactly — which is the value
  // to set if a deployment wants no part of it.
  //
  // 3 is small on purpose. The band exists so a place that has ARRIVED is not
  // an operator-reset event; it is not there to let a skill push repeatedly. A
  // skill that cannot get out of the band in three attempts is not grazing the
  // receptacle, and the fourth refusal latches.
  this->declare_parameter<std::int64_t>("place_advisory_max_consecutive", 3);

  // Distance-graded velocity scaling (#188). Width of the band ABOVE each
  // check's own gate margin in which an accepted chunk is slowed rather than
  // passed at full rate. **0 disables the mechanism entirely and reproduces
  // today's behaviour bit-for-bit**, which is the default: this changes an
  // enforcement surface, and the A/B five-round battery is what earns it a
  // non-zero value, not this parameter's existence.
  //
  // The scale is `exp(k · (slack − proximity))`, so it is exactly 1.0 at the
  // top of the band and `exp(−k · proximity)` at the margin itself. With the
  // suggested 0.05 m / 20.0 that is 1.00 → 0.37 across the band.
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

  // Phase 3 — predictive Cartesian look-ahead. The EE collision-link
  // index lets the kernel build the arm Jacobian and reconstruct where a
  // CARTESIAN_DELTA chunk's EE deltas drive the arm; <0 (default) leaves
  // predictive Cartesian disabled (reactive measured-config check only). The
  // launch derives the index from the robot's end-effector frame; lambda damps
  // the DLS solve near singularities; margin_growth inflates the collision
  // margin after each additional look-ahead step to bound accumulated
  // linearization/DLS residual (the first predicted step uses the configured
  // collision margin);
  // max_steps caps the look-ahead (0 = every row, last step always included).
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
  // `sim_e2e.launch.py` populates each field from
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
  q_check_.assign(ndof, 0.0);
  q_fk_.assign(ndof, 0.0);
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

  joint_name_to_dof_.clear();
  for (std::size_t i = 0; i < collision_joint_names_.size() && i < ndof; ++i) {
    joint_name_to_dof_.emplace(collision_joint_names_[i], static_cast<int>(i));
  }
  collision_fk_dofs_.clear();
  for (const int d : collision_model_.dof_index) {
    if (d >= 0 && static_cast<std::size_t>(d) < ndof) {
      collision_fk_dofs_.push_back(d);
    }
  }
  const bool seed_ready = !collision_joint_names_.empty() && !collision_fk_dofs_.empty();
  if (self_collision_enabled_ || world_collision_enabled_ || world_voxel_enabled_) {
    RCLCPP_INFO(this->get_logger(),
                "velocity+cartesian collision: %s (joint_names=%zu, fk_dofs=%zu, dt=%gs, "
                "state_deadline=%gs)",
                seed_ready ? "armed"
                           : "INACTIVE (no collision_joint_names — velocity/cartesian "
                             "chunks will be passed by geometry; plumb the param to cover)",
                collision_joint_names_.size(), collision_fk_dofs_.size(), collision_seed_dt_s_,
                collision_state_deadline_s_);
  }

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
  if ((self_collision_enabled_ || world_collision_enabled_ || world_voxel_enabled_ ||
       attached_collision_enabled_) &&
      !collision_joint_names_.empty()) {
    rclcpp::QoS js_qos(rclcpp::KeepLast(1));
    js_qos.best_effort();
    js_qos.durability_volatile();
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", js_qos,
        std::bind(&SafetyKernelLifecycleNode::on_joint_state, this, std::placeholders::_1));
  }

  if (world_collision_enabled_) {
    rclcpp::QoS world_qos(rclcpp::KeepLast(1));
    world_qos.reliable();
    world_qos.durability_volatile();
    world_sub_ = this->create_subscription<openral_msgs::msg::WorldCollision>(
        "/openral/world_collision", world_qos,
        std::bind(&SafetyKernelLifecycleNode::on_world_collision, this, std::placeholders::_1));
  }
  if (world_voxel_enabled_) {
    rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
    voxel_qos.reliable();
    voxel_qos.durability_volatile();
    voxel_sub_ = this->create_subscription<openral_msgs::msg::OccupancyVoxels>(
        "/openral/world_voxels", voxel_qos,
        std::bind(&SafetyKernelLifecycleNode::on_world_voxels, this, std::placeholders::_1));
  }
  if (attached_collision_enabled_) {
    // WorldStateStamped is published RELIABLE+VOLATILE+KL=1 at 30 Hz; the kernel
    // only consumes the bounded attached_objects array (grasped payloads).
    rclcpp::QoS world_state_qos(rclcpp::KeepLast(1));
    world_state_qos.reliable();
    world_state_qos.durability_volatile();
    world_state_sub_ = this->create_subscription<openral_msgs::msg::WorldStateStamped>(
        "/openral/world_state_fast", world_state_qos,
        std::bind(&SafetyKernelLifecycleNode::on_world_state, this, std::placeholders::_1));
    rclcpp::QoS attachment_ack_qos(rclcpp::KeepLast(1));
    attachment_ack_qos.reliable();
    attachment_ack_qos.transient_local();
    attachment_applied_pub_ = this->create_publisher<std_msgs::msg::UInt64>(
        "/openral/attachment_state_applied", attachment_ack_qos);
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
  estop_reset_srv_.reset();
  candidate_sub_.reset();
  estop_sub_.reset();
  world_sub_.reset();
  voxel_sub_.reset();
  world_state_sub_.reset();
  joint_state_sub_.reset();
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

  const auto result = validate(view, envelope_);
  if (result) {
    // Geometric collision over the chunk horizon (self + world).
    // Runs only for absolute joint-position chunks (the rows are full joint
    // configs FK can place). Allocation-free: FK reuses the pre-sized scratch.
    const bool geom_enabled = self_collision_enabled_ || world_collision_enabled_ ||
                              world_voxel_enabled_ || attached_collision_enabled_;
    const auto mode = static_cast<ControlMode>(view.control_mode);
    // Smallest clearance ABOVE its own gate margin seen anywhere in this
    // chunk's sweep (#188). Slack rather than raw distance because every check
    // carries a different margin and the predictive steps inflate theirs with
    // look-ahead depth — slack is the only margin-agnostic way to compare them.
    //
    // Pairs BELOW their margin that did not trip are deliberately excluded:
    // not tripping there means the pair is exempted (an attached payload's
    // attach-time contact baseline, an ACM row, a live place allowance), and an
    // exemption says that contact is allowed. Letting it drive the scale would
    // make a robot that is legitimately resting a payload on a shelf crawl for
    // as long as it holds it.
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
      if (world_collision_enabled_) {
        const bool fresh = world_received_ && !world_overflow_ &&
                           (this->now() - world_stamp_).seconds() <= world_collision_deadline_s_;
        if (!fresh) {
          unavailable(world_overflow_ ? "world_overflow" : "world_unavailable",
                      world_overflow_ ? openral_msgs::msg::SafetyStatus::DROP_WORLD_OVERFLOW
                                      : openral_msgs::msg::SafetyStatus::DROP_WORLD_UNAVAILABLE);
          return;
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
      const auto world_label = [this](int idx) -> std::string {
        if (idx >= 0 && static_cast<std::size_t>(idx) < world_labels_.size() &&
            !world_labels_[static_cast<std::size_t>(idx)].empty()) {
          return world_labels_[static_cast<std::size_t>(idx)];
        }
        return std::string("world_") + std::to_string(idx);
      };
      const auto attached_label = [this](int idx) -> std::string {
        if (idx >= 0 && static_cast<std::size_t>(idx) < attached_labels_.size() &&
            !attached_labels_[static_cast<std::size_t>(idx)].empty()) {
          return std::string("attached:") + attached_labels_[static_cast<std::size_t>(idx)];
        }
        return std::string("attached_") + std::to_string(idx);
      };
      // Report one collision hit. `hit` supplies BOTH the identity and the
      // distance, so the E-stop evidence always describes a single geometry
      // pair: `hit.min_distance` is that pair's own surface distance, never
      // the sweep-wide minimum. The sweep minimum — which may belong to a pair
      // the check exempted (an attached payload's attach-time contact
      // baseline) — is logged under its own `sweep_min_distance_m` key and its
      // own span attribute, and deliberately stays out of the evidence
      // payload's `min_distance_m`.
      //
      // The evidence also carries `q_fk_` — the configuration the hit was
      // measured at. `report` is only ever reached from `check_config`, right
      // after that config was FK'd, so this is the geometry the verdict is
      // about and nothing else can be. A predicted step's configuration is
      // reconstructible from no other artifact (it depends on the kernel's own
      // DLS Jacobian, its lambda and its dt), and the drawer-opening stop that
      // opened this issue was adjudicated against the *measured* joints, where
      // the two links it named sit 53 mm apart.
      const auto report = [&](const char* kind, const std::string& a, const std::string& b,
                              int step, const CollisionHit& hit) {
        ++chunks_dropped_;
        // The advisory band (#176): a declared payload's own receptacle contact
        // is refused, not latched — the chunk is dropped and the skill may try
        // again, instead of the operator having to call /openral/estop_reset
        // over a contact the ground truth measures in single millimetres.
        //
        // Bounded three ways, and any of them failing gives today's stop:
        // `hit.advisory` is only ever set for an attached payload inside its
        // own live declaration and within one voxel of the approach allowance;
        // the run of refusals is capped; and every other check in this function
        // reaches the same `report` with `advisory == false`.
        if (hit.advisory && advisory_refusals_ < place_advisory_max_consecutive_) {
          ++advisory_refusals_;
          last_drop_reason_ = "collision_advisory";
          publish_collision_failure(*msg, kind, a, b, step, hit.min_distance, q_fk_);
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
        publish_collision_failure(*msg, kind, a, b, step, hit.min_distance, q_fk_);
        set_safety_status(true, openral_msgs::msg::SafetyStatus::KIND_COLLISION, kind,
                          msg->rskill_id, msg->trace_id);
        std_msgs::msg::Empty estop_msg;
        estop_pub_->publish(estop_msg);
        // `place_allowance_active` is disclosure, never justification: the stop
        // happened, and the distance quoted is the pair's true one. It says the
        // margin that pair was gated against had been reduced by a live place
        // declaration, so an incident review can tell that case apart from an
        // ordinary stop without re-deriving it (CLAUDE.md §1.4).
        RCLCPP_ERROR(this->get_logger(),
                     "safety.collision kind=%s a=%s b=%s step=%d min_distance_m=%g "
                     "sweep_min_distance_m=%g mode=%u rskill_id=%s place_allowance_active=%d "
                     "place_target=%s",
                     kind, a.c_str(), b.c_str(), step, hit.min_distance, hit.sweep_min_distance,
                     static_cast<unsigned>(view.control_mode), msg->rskill_id.c_str(),
                     static_cast<int>(hit.place_allowance_active),
                     hit.place_allowance_active ? place_declaration_target_.c_str() : "");
        span->SetAttribute("safety.severity", "violation");
        span->SetAttribute("safety.drop_reason", "collision");
        span->SetAttribute("safety.collision_mode", static_cast<int64_t>(view.control_mode));
        span->SetAttribute("safety.violation_value", hit.min_distance);
        span->SetAttribute("safety.sweep_min_distance_m", hit.sweep_min_distance);
        span->SetAttribute("safety.place_allowance_active", hit.place_allowance_active);
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
      // FK `q` then run the enabled checks at `margin + extra_margin` (extra>0 for
      // predictive steps, inflating with look-ahead depth). report + return true
      // on the first hit. `q` is a position row, the measured seed, or a
      // predicted Cartesian config.
      // Fold one check's sweep-wide minimum into the chunk's smallest slack
      // (#188). `m` is the margin THAT check was gated against, including the
      // predictive look-ahead inflation, so the comparison is like-for-like.
      const auto note_slack = [&](const CollisionHit& h, double m) {
        const double slack = h.sweep_min_distance - m;
        if (slack >= 0.0 && slack < collision_min_slack_m) {
          collision_min_slack_m = slack;
        }
      };
      const auto check_config = [&](const double* q, int step, double extra_margin = 0.0) -> bool {
        fk_config(q);
        if (self_collision_enabled_) {
          const auto hit = check_self_collision(collision_model_, collision_scratch_,
                                                self_collision_margin_m_ + extra_margin);
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
            report("self", link_name(hit.link_a), link_name(hit.link_b), step, hit);
            return true;
          }
        }
        if (world_collision_enabled_) {
          const auto hit = check_world_collision(collision_model_, collision_scratch_, world_model_,
                                                 world_collision_margin_m_ + extra_margin);
          note_slack(hit, world_collision_margin_m_ + extra_margin);
          if (hit.hit) {
            report("world", link_name(hit.link_a), world_label(hit.link_b), step, hit);
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
          const auto hit = check_voxel_collision(collision_model_, collision_scratch_, voxel_grid_,
                                                 world_voxel_margin_m_ + extra_margin,
                                                 collision_scale_proximity_m_);
          note_slack(hit, world_voxel_margin_m_ + extra_margin);
          if (hit.hit) {
            report("world", link_name(hit.link_a),
                   std::string("voxel_") + std::to_string(hit.link_b), step, hit);
            return true;
          }
        }
        if (attached_collision_enabled_ && attached_model_.n_objects > 0) {
          // Grasped payloads (ADR-0092): check the attach-link-composed payload
          // geometry against world obstacles, occupancy voxels, and the robot's
          // own links (except the attach link + explicit touch links). The
          // collision_kind stays "self"/"world" to satisfy CollisionEvidence;
          // the attach:<object_id> label marks it as a payload hit.
          const double amargin = attached_collision_margin_m_ + extra_margin;
          if (world_collision_enabled_) {
            const auto hit = check_attached_world_collision(
                collision_model_, attached_model_, collision_scratch_, world_model_, amargin);
            note_slack(hit, amargin);
            if (hit.hit) {
              report("world", attached_label(hit.link_a), world_label(hit.link_b), step, hit);
              return true;
            }
          }
          // Legacy contact phase: without an attestation the kernel still can
          // not tell a legitimate support contact from a real one, so it keeps
          // skipping the predicted Cartesian steps. With a live witness it does
          // not need to — the exemption is pose-dependent and bounded — so the
          // predicted steps are checked too. That is strictly more checking.
          const bool contact_constrained_prediction =
              is_cartesian && attached_contact_active_ && step >= 0 && support_witness_live_ == 0;
          if (world_voxel_enabled_ && !contact_constrained_prediction) {
            const auto hit = check_attached_voxel_collision(
                collision_model_, attached_model_, collision_scratch_, voxel_grid_, amargin,
                collision_scale_proximity_m_);
            note_slack(hit, amargin);
            if (hit.hit) {
              // ADR-0098: when the declared target's own geometry adjudicated
              // the pair, `hit.min_distance` is the distance to that BODY, not
              // to the cell `link_b` indexes — so the evidence has to name the
              // body. Quoting one geometry's distance under another's identity
              // is precisely the failure `CollisionHit` forbids, and precisely
              // what #187 landed to stop the evidence doing.
              const std::string other =
                  hit.place_target_adjudicated
                      ? "place:" + place_declaration_target_ + "#" + std::to_string(hit.link_b)
                      : std::string("voxel_") + std::to_string(hit.link_b);
              report("world", attached_label(hit.link_a), other, step, hit);
              return true;
            }
          }
          const auto hit = check_attached_self_collision(collision_model_, attached_model_,
                                                         collision_scratch_, amargin);
          note_slack(hit, amargin);
          if (hit.hit) {
            report("self", attached_label(hit.link_a), link_name(hit.link_b), step, hit);
            return true;
          }
        }
        return false;
      };

      if (is_position) {
        // Each row is a full joint configuration FK can place directly.
        for (std::uint16_t s = 0; s < view.horizon; ++s) {
          const double* row = view.flat_data + static_cast<std::size_t>(s) * robot_ndof;
          if (check_config(row, static_cast<int>(s))) {
            return;
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

    // Distance-graded slowdown (#188). The chunk has ALREADY been accepted —
    // this never turns an accept into a drop, and never a drop into an accept.
    // It only decides how fast the accepted motion is allowed to be executed.
    //
    // Only rate-shaped modes are scalable, and the distinction is not
    // stylistic: multiplying an ABSOLUTE target (JOINT_POSITION,
    // CARTESIAN_POSE, JOINT_TRAJECTORY) by 0.5 does not halve the speed, it
    // commands a pose half way to the origin — a large unrequested motion, in
    // an unpredictable direction, dressed up as a safety measure. Shrinking
    // those toward the measured configuration would be the correct form and
    // needs a fresh seed that position mode deliberately does not require
    // today, so absolute modes are left alone. Gripper / dex-hand / composite
    // rows are not motion rates either (a scaled grasp is a dropped object, a
    // scaled mode flag is a corrupted multiplexer). BODY_TWIST is excluded on
    // evidence, not convenience: FK zeroes the base dofs, so nothing here
    // measures where the BASE is going, and slowing it for an arm-side
    // clearance would be a number applied to the wrong body (the base's own
    // proximity gate is Nav2's collision monitor, #186).
    const bool rate_shaped = mode == ControlMode::kJointVelocity ||
                             mode == ControlMode::kCartesianDelta ||
                             mode == ControlMode::kCartesianTwist;
    const double scale = (geom_enabled && rate_shaped)
                             ? velocity_scale_for(collision_min_slack_m)
                             : 1.0;
    if (scale < 1.0) {
      // Scaling DOWN a rate keeps every envelope bound it already satisfied
      // (|s·v| <= |v| for s in [0,1]), so the scaled chunk needs no
      // re-validation, and it travels no further than the swept volume the
      // predictive check just cleared. Strictly more conservative, both ways.
      scaled_chunk_ = *msg;
      const std::size_t stride = view.n_dof;
      // A Cartesian row is a 6-vector twist [vx,vy,vz,wx,wy,wz] inside a row of
      // `n_dof`; a velocity row is velocities all the way across.
      const std::size_t scalable = (mode == ControlMode::kJointVelocity)
                                       ? stride
                                       : std::min<std::size_t>(stride, 6);
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
  const std::size_t n = std::min(msg->name.size(), msg->position.size());
  for (std::size_t i = 0; i < n; ++i) {
    const auto it = joint_name_to_dof_.find(msg->name[i]);
    if (it == joint_name_to_dof_.end()) {
      continue;
    }
    const std::size_t d = static_cast<std::size_t>(it->second);
    if (d < q_meas_.size()) {
      q_meas_[d] = msg->position[i];
      q_meas_seen_[d] = true;
    }
  }
  q_meas_received_ = true;
  q_meas_stamp_ = this->now();
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

void SafetyKernelLifecycleNode::on_external_estop(const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  if (!fault_latch_) {
    fault_latch_ = true;
    last_estop_at_ = std::chrono::steady_clock::now();
    last_drop_reason_ = "external_estop";
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
  // Standing place-declaration state. The refusal logs are transition-gated, so
  // this heartbeat is what makes a persistent refusal (or a live allowance)
  // observable without re-warning at the attachment rate.
  std::string place_region_state{"-"};
  if (place_region_.valid) {
    place_region_state = (place_declaration_live() ? "live:" : "expired:") +
                         place_declaration_target_ + ":geom=" +
                         std::to_string(place_region_.n_geometry);
  } else if (!place_region_refusal_reason_.empty()) {
    place_region_state = place_region_refusal_reason_ + ":" + place_region_refusal_target_;
  }
  add_kv("place_region", place_region_state);
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
    oss << R"({"kind":"force","joint_or_ee":")"
        << (v.joint_index == 0xFFFF ? std::string{"ee"}
                                    : std::string{"joint_"} + std::to_string(v.joint_index))
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
    const std::string ee_name = is_cartesian
                                    ? std::string{"end_effector"}
                                    : std::string{"joint_"} + std::to_string(v.joint_index);
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
    oss << R"({"kind":"workspace","ee_name":")" << ee_name << R"(","measured_xyz":[)" << mx << ","
        << my << "," << mz << R"(],"box_min":[)" << box_min_x << R"(,0.0,0.0],"box_max":[)"
        << box_max_x << R"(,0.0,0.0]})";
    break;
  }
  case ViolationKind::kController:
  default: {
    // ControllerEvidence: controller_name, state, detail.
    const std::string state = (v.sub == ControllerSubKind::kNanInAction)    ? "nan_in_action"
                              : (v.sub == ControllerSubKind::kNdofMismatch) ? "ndof_mismatch"
                              : (v.sub == ControllerSubKind::kDimMismatch)  ? "dim_mismatch"
                              : (v.sub == ControllerSubKind::kInvalidScale) ? "invalid_scale"
                              : (v.sub == ControllerSubKind::kEnvelopeUnconfigured)
                                  ? "envelope_unconfigured"
                                  : "controller_error";
    std::ostringstream detail;
    detail << "field=" << v.field << " joint=" << v.joint_index << " value=" << v.offending_value
           << " limit=" << v.limit_value;
    oss << R"({"kind":"controller","controller_name":"openral_safety_kernel","state":")" << state
        << R"(","detail":")" << detail.str() << R"("})";
    break;
  }
  }
  trigger.evidence_json = oss.str();
  failure_pub_->publish(trigger);
}

bool SafetyKernelLifecycleNode::load_collision_model(std::string& error) {
  // Reset to the disabled state so a re-configure can never leak a stale model.
  collision_model_ = CollisionModel{};
  collision_link_names_.clear();
  collision_scratch_.link_world.clear();
  collision_required_dof_ = 0;
  self_collision_enabled_ = this->get_parameter("self_collision_enabled").as_bool();

  // World-collision config (shares the robot collision model — the same link
  // capsules are checked against world obstacles).
  world_collision_enabled_ = this->get_parameter("world_collision_enabled").as_bool();
  world_collision_margin_m_ = this->get_parameter("world_collision_margin_m").as_double();
  world_collision_deadline_s_ =
      this->get_parameter("world_collision_deadline_ms").as_double() / 1000.0;
  world_collision_max_primitives_ =
      static_cast<std::size_t>(this->get_parameter("world_collision_max_primitives").as_int());
  world_received_ = false;
  world_overflow_ = false;
  world_model_.capsules.clear();
  world_labels_.clear();

  // Voxel (dense occupancy grid) config. Pre-size the occupancy buffer once so
  // the subscription callback never reallocates and the view pointer is stable.
  world_voxel_enabled_ = this->get_parameter("world_voxel_enabled").as_bool();
  world_voxel_margin_m_ = this->get_parameter("world_voxel_margin_m").as_double();
  world_voxel_deadline_s_ = this->get_parameter("world_voxel_deadline_ms").as_double() / 1000.0;
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
  place_declaration_stamp_ns_ = 0;
  place_declaration_timeout_s_ = 0.0;
  place_declaration_target_.clear();
  voxel_frame_id_.clear();
  attached_model_ = AttachedModel{};
  attached_model_.objects.assign(attached_max_objects_, AttachedObject{});
  attached_model_.primitives.assign(attached_max_primitives_, AttachedPrimitive{});
  attached_model_.touch_links.assign(attached_max_touch_links_, 0);
  attached_labels_.assign(attached_max_objects_, std::string{});
  attached_ingest_scratch_.clear();
  attached_ingest_scratch_.reserve(attached_max_objects_);
  // Sized once, here, and never resized again: `place_region_.geometry` is a
  // view into this buffer and the hot path reads it (ADR-0098).
  place_geometry_.assign(kMaxPlaceTargetPrimitives, AttachedPrimitive{});
  place_geometry_scratch_.clear();
  place_geometry_scratch_.reserve(kMaxPlaceTargetPrimitives);

  // The robot collision model is needed for any geometric check; skip loading
  // only when all of them are disabled.
  if (!self_collision_enabled_ && !world_collision_enabled_ && !world_voxel_enabled_ &&
      !attached_collision_enabled_) {
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

  CollisionModel m;
  m.n_links = n_links;
  m.parent.resize(n_links);
  m.joint_kind.resize(n_links);
  m.dof_index.resize(n_links);
  m.origin.resize(n_links);
  m.axis.resize(n_links);
  for (std::size_t i = 0; i < n_links; ++i) {
    m.parent[i] = static_cast<int>(parent[i]);
    m.joint_kind[i] = static_cast<JointKind>(static_cast<std::uint8_t>(kind[i]));
    m.dof_index[i] = static_cast<int>(dof[i]);
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
    const std::string& link_a, const std::string& link_b, int horizon_step, double min_distance,
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
  oss << R"({"kind":"collision","collision_kind":")" << collision_kind << R"(","link_a":")"
      << link_a << R"(","link_b_or_object":")" << link_b << R"(","horizon_step":)" << horizon_step
      << R"(,"min_distance_m":)" << min_distance << R"(,"joint_positions_rad":[)";
  for (std::size_t i = 0; i < joint_positions.size(); ++i) {
    if (i != 0) {
      oss << ',';
    }
    oss << joint_positions[i];
  }
  oss << "]}";
  trigger.evidence_json = oss.str();
  failure_pub_->publish(trigger);
}

void SafetyKernelLifecycleNode::on_world_collision(
    const openral_msgs::msg::WorldCollision::SharedPtr msg) {
  if (msg == nullptr) {
    return;
  }
  const std::size_t n = msg->radius.size();
  // Shape + capacity validation. Over-capacity or malformed → fail closed:
  // mark the world invalid so the next chunk is dropped until a good one lands.
  if (msg->half_length.size() != n || msg->origin_xyzrpy.size() != 6 * n ||
      n > world_collision_max_primitives_) {
    world_overflow_ = true;
    world_received_ = true;
    world_stamp_ = this->now();
    return;
  }
  world_overflow_ = false;
  world_model_.capsules.resize(n);
  world_labels_.assign(n, std::string{});
  for (std::size_t i = 0; i < n; ++i) {
    world_model_.capsules[i].radius = msg->radius[i];
    world_model_.capsules[i].half_length = msg->half_length[i];
    world_model_.capsules[i].origin =
        transform_from_xyz_rpy(msg->origin_xyzrpy[6 * i + 0], msg->origin_xyzrpy[6 * i + 1],
                               msg->origin_xyzrpy[6 * i + 2], msg->origin_xyzrpy[6 * i + 3],
                               msg->origin_xyzrpy[6 * i + 4], msg->origin_xyzrpy[6 * i + 5]);
    if (i < msg->object_id.size()) {
      world_labels_[i] = msg->object_id[i];
    }
  }
  world_received_ = true;
  world_stamp_ = this->now();
}

void SafetyKernelLifecycleNode::on_world_voxels(
    const openral_msgs::msg::OccupancyVoxels::SharedPtr msg) {
  if (msg == nullptr) {
    return;
  }
  const std::size_t cells = static_cast<std::size_t>(msg->size_x) * msg->size_y * msg->size_z;
  // The grid's lattice is the source map's, so `orientation` is load-bearing
  // geometry, not metadata. An unset field is the all-zero quaternion, which is
  // not a rotation: reading it as identity would place every obstacle somewhere
  // the robot is not, and a world check against a misplaced map is fail-OPEN.
  // Refuse it exactly as an over-large or malformed grid is refused.
  const double quat_norm2 =
      msg->orientation.x * msg->orientation.x + msg->orientation.y * msg->orientation.y +
      msg->orientation.z * msg->orientation.z + msg->orientation.w * msg->orientation.w;
  if (msg->occupancy.size() != cells || cells > world_voxel_max_cells_ || msg->resolution <= 0.0 ||
      !std::isfinite(quat_norm2) || std::fabs(quat_norm2 - 1.0) > 1e-6) {
    voxel_overflow_ = true;
    voxel_received_ = true;
    voxel_stamp_ = this->now();
    return;
  }
  voxel_overflow_ = false;
  std::copy(msg->occupancy.begin(), msg->occupancy.end(), voxel_occupancy_.begin());
  voxel_grid_.occupancy = voxel_occupancy_.data();
  voxel_grid_.pose = transform_from_translation_quat(
      msg->origin.x, msg->origin.y, msg->origin.z, msg->orientation.x, msg->orientation.y,
      msg->orientation.z, msg->orientation.w);
  voxel_grid_.resolution = msg->resolution;
  voxel_grid_.sx = static_cast<int>(msg->size_x);
  voxel_grid_.sy = static_cast<int>(msg->size_y);
  voxel_grid_.sz = static_cast<int>(msg->size_z);
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
  }
  voxel_received_ = true;
  voxel_stamp_ = this->now();
}

void SafetyKernelLifecycleNode::on_world_state(
    const openral_msgs::msg::WorldStateStamped::SharedPtr msg) {
  if (msg == nullptr) {
    return;
  }
  // Parse the wire attachments into the by-name ingest scratch, then validate +
  // resolve into the fixed-capacity attached model. Any malformed shape,
  // capacity overflow, or unknown attach/touch link fails closed: the model is
  // emptied and the world state is marked invalid so the next candidate action
  // is dropped until a clean message lands (never a silently-unchecked payload).
  const auto clock_type = this->get_clock()->get_clock_type();
  const bool stamp_valid = msg->attachment_stamp_ns > 0;
  const rclcpp::Time producer_stamp(stamp_valid ? msg->attachment_stamp_ns : 0, clock_type);
  const auto fail_closed = [this, producer_stamp]() {
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
  };

  const auto& wire = msg->attached_objects;
  if (!stamp_valid) {
    fail_closed();
    return;
  }
  if (wire.size() > attached_max_objects_) {
    fail_closed();
    return;
  }
  // Total-primitive cap across every attached object (early fail-closed before
  // the by-name ingest; ingest_attached_objects re-checks incrementally).
  std::size_t total_primitives = 0;
  for (const auto& obj : wire) {
    total_primitives += obj.primitives.size();
  }
  if (total_primitives > attached_max_primitives_) {
    fail_closed();
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
        fail_closed();
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
    in.primitives.clear();
    in.primitives.reserve(obj.primitives.size());
    for (const auto& prim : obj.primitives) {
      AttachedPrimitiveInput pin;
      // Unknown shape tag or too few dimensions → fail closed (an unrecognised
      // payload is not safe to ignore).
      if (!decode_attached_primitive(prim, pin)) {
        fail_closed();
        return;
      }
      in.primitives.push_back(pin);
    }
    attached_ingest_scratch_.push_back(std::move(in));
  }

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
    return;
  }
  // Labels parallel the accepted objects (object_id for evidence).
  for (std::size_t i = 0; i < attached_model_.n_objects; ++i) {
    attached_labels_[i] = wire[i].object_id;
  }
  // Arm a support-contact witness only when World State attests a genuinely NEW
  // one. The attachment set is heartbeated, so keying off message arrival would
  // silently resurrect an exemption the kernel had already killed on
  // separation. The key is (object id, support id, producer stamp): a regrasp
  // or a re-contact produces a fresh stamp and re-arms honestly; a republished
  // snapshot does not.
  for (std::size_t i = 0; i < support_witness_keys_.size(); ++i) {
    const auto bit = static_cast<std::uint8_t>(1U << i);
    if (i >= attached_model_.n_objects || !attached_model_.objects[i].has_support_witness) {
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
      place_region_refusal_reason_.clear();
      place_region_refusal_target_.clear();
    } else {
      attached_contact_snapshot_pending_ = true;
      attached_contact_active_ = false;
    }
  }
  // The place declaration rides the same snapshot as the payload it is scoped
  // to, so it is resolved here, against the objects that were just accepted.
  ingest_place_declaration(*msg);
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
  place_declaration_stamp_ns_ = 0;
  place_declaration_timeout_s_ = 0.0;
  place_declaration_target_.clear();

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
  place_declaration_stamp_ns_ = declaration.stamp_ns;
  place_declaration_timeout_s_ = declaration.timeout_s;
  if (!declaration.active) {
    announce_dropped("retracted");
    return;
  }
  if (!declaration.region_valid) {
    // The common, correct case for dispatch's own publication and for every real
    // deployment today: a declaration with no producer-measured region. It still
    // gates the place witness; it buys no approach allowance.
    announce_dropped("no_region");
    return;
  }
  const auto& region = declaration.region;
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
  // Which carried payload the allowance follows. An empty object_id is the
  // direct-dispatch case ("whichever payload is carried"), which the producer
  // resolves at attach time; the kernel maps it onto every accepted object.
  std::uint8_t object_mask = 0;
  const std::size_t objects = std::min<std::size_t>(attached_model_.n_objects, 8);
  for (std::size_t i = 0; i < objects; ++i) {
    if (declaration.object_id.empty() || attached_labels_[i] == declaration.object_id) {
      object_mask = static_cast<std::uint8_t>(object_mask | (1U << i));
    }
  }
  const Transform pose = transform_from_translation_quat(
      region.pose.position.x, region.pose.position.y, region.pose.position.z,
      region.pose.orientation.x, region.pose.orientation.y, region.pose.orientation.z,
      region.pose.orientation.w);
  const Vec3 half{region.half_extents.x, region.half_extents.y, region.half_extents.z};
  PlaceRegionStatus status = ingest_place_region(pose, half, object_mask, place_region_);
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
      status = ingest_place_target_geometry(place_geometry_scratch_, place_geometry_, place_region_);
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
         elapsed_ns <= static_cast<std::int64_t>(place_declaration_timeout_s_ * 1e9);
}

}  // namespace openral_safety_kernel

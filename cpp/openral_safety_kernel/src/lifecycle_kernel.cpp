// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — SafetyKernelLifecycleNode source.

#include "openral_safety_kernel/lifecycle_kernel.hpp"

#include "openral_safety_kernel/otel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <opentelemetry/common/attribute_value.h>
#include <opentelemetry/context/runtime_context.h>
#include <opentelemetry/trace/scope.h>
#include <opentelemetry/trace/span.h>
#include <opentelemetry/trace/tracer.h>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rclcpp/qos.hpp>

namespace openral_safety_kernel {

namespace {

rclcpp::QoS chunk_qos() {
  // ADR-0028b — the openral slot dispatcher publishes N chunks per
  // policy tick on /openral/candidate_action (arm CARTESIAN_DELTA +
  // gripper GRIPPER_POSITION + optional base BODY_TWIST). KEEP_LAST=1
  // on the subscriber side coalesces back-to-back publishes inside
  // the same callback batch: only the last slot's chunk survives, so
  // in deploy_sim the arm freezes while the gripper keeps streaming
  // because that's the LAST published slot per tick. Depth=10 matches
  // CLAUDE.md §3's "safety/e-stop = KEEP_LAST=10" guidance and is the
  // minimum that survives slot fan-out at 30 Hz tick rate.
  rclcpp::QoS q(rclcpp::KeepLast(10));
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

  // ADR-0020 PR-K (2026-05-24) — parameter-based envelope source. The
  // Python `sim_e2e.launch.py` unpacks `robots/<id>/robot.yaml` via
  // Pydantic, calls
  // `openral_safety.envelope_loader.kernel_params_from_envelope`, and
  // forwards each field as a ROS parameter here. There is exactly one
  // transport: ROS parameters. The flat-YAML `envelope_file:=PATH`
  // path the kernel had pre-PR-K is gone.
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

  // ADR-0030 — self-collision model. Disabled unless the launch emits a
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
  this->declare_parameter<std::vector<std::int64_t>>("collision_allowed_pairs",
                                                     std::vector<std::int64_t>{});
  this->declare_parameter<std::vector<std::string>>("collision_link_names",
                                                    std::vector<std::string>{});

  // ADR-0030 world phase — world-obstacle collision check (opt-in). Obstacles
  // arrive on /openral/world_collision in the robot base frame.
  this->declare_parameter<bool>("world_collision_enabled", false);
  this->declare_parameter<double>("world_collision_margin_m", 0.0);
  this->declare_parameter<double>("world_collision_deadline_ms", 500.0);
  this->declare_parameter<std::int64_t>("world_collision_max_primitives", 64);

  // ADR-0030 voxel phase — dense occupancy-grid world check (octomap path).
  this->declare_parameter<bool>("world_voxel_enabled", false);
  this->declare_parameter<double>("world_voxel_margin_m", 0.0);
  this->declare_parameter<double>("world_voxel_deadline_ms", 500.0);
  this->declare_parameter<std::int64_t>("world_voxel_max_cells", 262144);

  // ADR-0040 — measured joint-state seed for non-position collision checks.
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
  // ADR-0040 — dof indices of the planar mobile-base joints (description
  // base_joints). They are zeroed before the base-relative geometric FK so the
  // arm is placed in the base_link frame the world/voxel grid lives in;
  // otherwise FK applies the base's world pose and the arm sits metres outside
  // the local map. Empty for fixed-base arms (no-op).
  this->declare_parameter<std::vector<std::int64_t>>("collision_base_dofs",
                                                     std::vector<std::int64_t>{});

  // ADR-0040 Phase 3 — predictive Cartesian look-ahead. The EE collision-link
  // index lets the kernel build the arm Jacobian and reconstruct where a
  // CARTESIAN_DELTA chunk's EE deltas drive the arm; <0 (default) leaves
  // predictive Cartesian disabled (reactive measured-config check only). The
  // launch derives the index from the robot's end-effector frame; lambda damps
  // the DLS solve near singularities; margin_growth inflates the collision
  // margin per look-ahead step to bound the linearization/DLS residual;
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

  // Load envelope from ROS parameters (ADR-0020 PR-K). The Python
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

  // ADR-0030 — load the optional self-collision model. A malformed model when
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
  }

  // ADR-0040 — set up the measured joint-state seed used to reconstruct
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

  // ADR-0040 Phase 3 — predictive Cartesian scratch + params. dof_blocked_ marks
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
                "ADR-0040 velocity+cartesian collision: %s (joint_names=%zu, fk_dofs=%zu, dt=%gs, "
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
  diagnostics_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", rclcpp::QoS(rclcpp::KeepLast(1)));

  candidate_sub_ = this->create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos(),
      std::bind(&SafetyKernelLifecycleNode::on_candidate_action, this, std::placeholders::_1));
  estop_sub_ = this->create_subscription<std_msgs::msg::Empty>(
      "/openral/estop", estop_qos(),
      std::bind(&SafetyKernelLifecycleNode::on_external_estop, this, std::placeholders::_1));

  // ADR-0040 — subscribe /joint_states only when a geometric check is enabled
  // and the joint-name map is plumbed (otherwise there is nothing to seed).
  if ((self_collision_enabled_ || world_collision_enabled_ || world_voxel_enabled_) &&
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
  return rclcpp_lifecycle::LifecycleNode::on_activate(state);
}

SafetyKernelLifecycleNode::CallbackReturn
SafetyKernelLifecycleNode::on_deactivate(const rclcpp_lifecycle::State& state) {
  safe_pub_->on_deactivate();
  estop_pub_->on_deactivate();
  failure_pub_->on_deactivate();
  diagnostics_pub_->on_deactivate();
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
  joint_state_sub_.reset();
  q_meas_received_ = false;
  safe_pub_.reset();
  estop_pub_.reset();
  failure_pub_.reset();
  diagnostics_pub_.reset();
  envelope_loaded_ = false;
  fault_latch_ = false;
  chunks_passed_ = 0;
  chunks_dropped_ = 0;
  last_drop_reason_.clear();
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
  // in `trace_id` (ADR-0018 §6 — "OTel context is the truth; ROS fields
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
                                 {"safety.clamped", false},
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

  const auto result = validate(view, envelope_);
  if (result) {
    // ADR-0030 — geometric collision over the chunk horizon (self + world).
    // Runs only for absolute joint-position chunks (the rows are full joint
    // configs FK can place). Allocation-free: FK reuses the pre-sized scratch.
    const bool geom_enabled =
        self_collision_enabled_ || world_collision_enabled_ || world_voxel_enabled_;
    const auto mode = static_cast<ControlMode>(view.control_mode);
    const bool is_position = (mode == ControlMode::kJointPosition);
    // ADR-0040 — non-position chunks carry velocities / EE deltas, not joint
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
      const auto unavailable = [&](const char* reason) {
        ++chunks_dropped_;
        last_drop_reason_ = reason;
        Violation v;
        v.kind = ViolationKind::kController;
        v.set_field(reason);
        publish_failure_trigger(*msg, v);
        RCLCPP_WARN(this->get_logger(), "safety.world_unavailable reason=%s rskill_id=%s", reason,
                    msg->rskill_id.c_str());
        span->SetAttribute("safety.severity", "warn");
        span->SetAttribute("safety.drop_reason", reason);
        span->End();
      };
      // ADR-0040 — velocity/Cartesian reconstruction needs a fresh, complete
      // measured seed; fail-closed otherwise.
      if ((is_velocity || is_cartesian) && !measured_state_fresh()) {
        unavailable("state_unavailable");
        return;
      }
      if (world_collision_enabled_) {
        const bool fresh = world_received_ && !world_overflow_ &&
                           (this->now() - world_stamp_).seconds() <= world_collision_deadline_s_;
        if (!fresh) {
          unavailable(world_overflow_ ? "world_overflow" : "world_unavailable");
          return;
        }
      }
      if (world_voxel_enabled_) {
        const bool fresh = voxel_received_ && !voxel_overflow_ &&
                           (this->now() - voxel_stamp_).seconds() <= world_voxel_deadline_s_;
        if (!fresh) {
          unavailable(voxel_overflow_ ? "voxel_overflow" : "voxel_unavailable");
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
      const auto report = [&](const char* kind, const std::string& a, const std::string& b,
                              int step, double dist) {
        ++chunks_dropped_;
        last_drop_reason_ = "collision";
        fault_latch_ = true;
        last_estop_at_ = std::chrono::steady_clock::now();
        publish_collision_failure(*msg, kind, a, b, step, dist);
        std_msgs::msg::Empty estop_msg;
        estop_pub_->publish(estop_msg);
        RCLCPP_ERROR(this->get_logger(),
                     "safety.collision kind=%s a=%s b=%s step=%d min_distance_m=%g mode=%u "
                     "rskill_id=%s",
                     kind, a.c_str(), b.c_str(), step, dist,
                     static_cast<unsigned>(view.control_mode), msg->rskill_id.c_str());
        span->SetAttribute("safety.severity", "violation");
        span->SetAttribute("safety.drop_reason", "collision");
        span->SetAttribute("safety.collision_mode", static_cast<int64_t>(view.control_mode));
        span->SetAttribute("safety.violation_value", dist);
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
      // FK `q` then run the enabled checks at `margin + extra_margin` (extra>0 for
      // predictive steps, inflating with look-ahead depth). report + return true
      // on the first hit. `q` is a position row, the measured seed, or a
      // predicted Cartesian config.
      const auto check_config = [&](const double* q, int step, double extra_margin = 0.0) -> bool {
        fk_config(q);
        if (self_collision_enabled_) {
          const auto hit = check_self_collision(collision_model_, collision_scratch_,
                                                self_collision_margin_m_ + extra_margin);
          if (hit.hit) {
            report("self", link_name(hit.link_a), link_name(hit.link_b), step, hit.min_distance);
            return true;
          }
        }
        if (world_collision_enabled_) {
          const auto hit = check_world_collision(collision_model_, collision_scratch_, world_model_,
                                                 world_collision_margin_m_ + extra_margin);
          if (hit.hit) {
            report("world", link_name(hit.link_a), world_label(hit.link_b), step, hit.min_distance);
            return true;
          }
        }
        if (world_voxel_enabled_) {
          const auto hit = check_voxel_collision(collision_model_, collision_scratch_, voxel_grid_,
                                                 world_voxel_margin_m_ + extra_margin);
          if (hit.hit) {
            report("world", link_name(hit.link_a),
                   std::string("voxel_") + std::to_string(hit.link_b), step, hit.min_distance);
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
        // Predictive Cartesian (CARTESIAN_DELTA, ADR-0040 Phase 3): reconstruct
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
            const double twist[6] = {row[0], row[1], row[2], row[3], row[4], row[5]};
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
              const double extra = collision_predict_margin_growth_m_ * static_cast<double>(s + 1);
              if (check_config(q_predict_.data(), static_cast<int>(s), extra)) {
                return;
              }
            }
          }
        }
      }
    }

    safe_pub_->publish(*msg);
    ++chunks_passed_;
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
  // ADR-0040 Phase 1 — fold the measured positions into q_meas_ in the action's
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
  add_kv("last_drop_reason", last_drop_reason_.empty() ? "-" : last_drop_reason_);
  add_kv("envelope_loaded", envelope_loaded_ ? "true" : "false");
  add_kv("n_dof", std::to_string(envelope_.n_dof));
  arr.status.push_back(status);
  diagnostics_pub_->publish(arr);
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
        << my << "," << mz << R"(],"box_min":[)" << box_min_x << ",0.0,0.0],\"box_max\":["
        << box_max_x << ",0.0,0.0]}";
    // The escaped-quote line above is a c-string artefact of the raw
    // literal break; we use a real raw literal for the box_max key:
    // build the rest of the JSON without the embedded backslash.
    // Recompose cleanly:
    oss.str("");
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

  // The robot collision model is needed for any geometric check; skip loading
  // only when all of them are disabled.
  if (!self_collision_enabled_ && !world_collision_enabled_ && !world_voxel_enabled_) {
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
  const auto pairs = this->get_parameter("collision_allowed_pairs").as_integer_array();
  const auto names = this->get_parameter("collision_link_names").as_string_array();

  // Per-link arrays are sized to n_links; capsule arrays are sized to the
  // (independent) capsule count — a link may carry zero, one, or several.
  const std::size_t n_caps = cap_r.size();
  if (parent.size() != n_links || kind.size() != n_links || dof.size() != n_links ||
      origin.size() != 6 * n_links || axis.size() != 3 * n_links || names.size() != n_links ||
      cap_link.size() != n_caps || cap_h.size() != n_caps || cap_o.size() != 6 * n_caps ||
      pairs.size() % 2 != 0) {
    error = "collision_* array shapes disagree with collision_n_links / capsule count";
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
  for (std::size_t k = 0; k + 1 < pairs.size(); k += 2) {
    m.allowed_pairs.emplace_back(static_cast<int>(pairs[k]), static_cast<int>(pairs[k + 1]));
  }

  collision_model_ = std::move(m);
  collision_link_names_.assign(names.begin(), names.end());
  collision_scratch_.link_world.resize(n_links);
  return true;
}

void SafetyKernelLifecycleNode::publish_collision_failure(
    const openral_msgs::msg::ActionChunk& chunk, const char* collision_kind,
    const std::string& link_a, const std::string& link_b, int horizon_step, double min_distance) {
  openral_msgs::msg::FailureTrigger trigger;
  trigger.header.stamp = this->now();
  trigger.kind = openral_msgs::msg::FailureTrigger::KIND_COLLISION;
  trigger.severity = openral_msgs::msg::FailureTrigger::SEVERITY_ABORT;
  trigger.rskill_id = chunk.rskill_id;
  trigger.trace_id = chunk.trace_id;

  // Shape matches openral_core.CollisionEvidence (kind="collision").
  std::ostringstream oss;
  oss << R"({"kind":"collision","collision_kind":")" << collision_kind << R"(","link_a":")"
      << link_a << R"(","link_b_or_object":")" << link_b << R"(","horizon_step":)" << horizon_step
      << R"(,"min_distance_m":)" << min_distance << "}";
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
  if (msg->occupancy.size() != cells || cells > world_voxel_max_cells_ || msg->resolution <= 0.0) {
    voxel_overflow_ = true;
    voxel_received_ = true;
    voxel_stamp_ = this->now();
    return;
  }
  voxel_overflow_ = false;
  std::copy(msg->occupancy.begin(), msg->occupancy.end(), voxel_occupancy_.begin());
  voxel_grid_.occupancy = voxel_occupancy_.data();
  voxel_grid_.origin = Vec3{msg->origin.x, msg->origin.y, msg->origin.z};
  voxel_grid_.resolution = msg->resolution;
  voxel_grid_.sx = static_cast<int>(msg->size_x);
  voxel_grid_.sy = static_cast<int>(msg->size_y);
  voxel_grid_.sz = static_cast<int>(msg->size_z);
  voxel_received_ = true;
  voxel_stamp_ = this->now();
}

}  // namespace openral_safety_kernel

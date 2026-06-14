// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — SafetyKernelLifecycleNode: the rclcpp_lifecycle::LifecycleNode
// that owns /openral/{candidate_action,safe_action,estop,failure/safety}
// and the /openral/estop_reset service. Replaces the F5 Python pass-
// through behind the same topic contract.

#pragma once

#include "openral_safety_kernel/collision.hpp"
#include "openral_safety_kernel/envelope.hpp"
#include "openral_safety_kernel/otel.hpp"
#include "openral_safety_kernel/validator.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <sensor_msgs/msg/joint_state.hpp>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <openral_msgs/msg/action_chunk.hpp>
#include <openral_msgs/msg/failure_trigger.hpp>
#include <openral_msgs/msg/occupancy_voxels.hpp>
#include <openral_msgs/msg/world_collision.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace openral_safety_kernel {

/// Default cooldown between an estop publish and the first successful
/// /openral/estop_reset call. Mirrors the Python F5 default.
inline constexpr double kDefaultEstopResetCooldownSec = 0.5;

/// Default chunk-validation deadline. The validator p99 must come in
/// well under this on the reference host (≤1 ms target).
inline constexpr std::int64_t kDefaultChunkValidationDeadlineUs = 1000;

class SafetyKernelLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit SafetyKernelLifecycleNode(const std::string& node_name = "openral_safety_kernel",
                                     const rclcpp::NodeOptions& options = rclcpp::NodeOptions{});

  ~SafetyKernelLifecycleNode() override = default;
  SafetyKernelLifecycleNode(const SafetyKernelLifecycleNode&) = delete;
  SafetyKernelLifecycleNode& operator=(const SafetyKernelLifecycleNode&) = delete;
  SafetyKernelLifecycleNode(SafetyKernelLifecycleNode&&) = delete;
  SafetyKernelLifecycleNode& operator=(SafetyKernelLifecycleNode&&) = delete;

  // ── Lifecycle callbacks ────────────────────────────────────────────────────

  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

  // ── Inspection helpers (for tests only) ────────────────────────────────────

  bool fault_latched() const noexcept { return fault_latch_; }
  std::uint64_t chunks_passed() const noexcept { return chunks_passed_; }
  std::uint64_t chunks_dropped() const noexcept { return chunks_dropped_; }
  const EnvelopeIntersection& envelope() const noexcept { return envelope_; }
  bool self_collision_active() const noexcept { return self_collision_enabled_; }
  std::size_t collision_link_count() const noexcept { return collision_model_.n_links; }

private:
  // Topic callbacks.
  void on_candidate_action(const openral_msgs::msg::ActionChunk::SharedPtr msg);
  void on_external_estop(const std_msgs::msg::Empty::SharedPtr msg);
  void on_estop_reset(const std_srvs::srv::Trigger::Request::SharedPtr request,
                      const std_srvs::srv::Trigger::Response::SharedPtr response);

  // Diagnostics heartbeat (1 Hz).
  void publish_diagnostics();

  // Publish a FailureTrigger on /openral/failure/safety for `violation`.
  void publish_failure_trigger(const openral_msgs::msg::ActionChunk& chunk,
                               const Violation& violation);

  // ADR-0030 — load the self-collision model from ROS parameters (configure
  // time; allocation OK). Returns false with `error` set on a malformed model.
  bool load_collision_model(std::string& error);

  // Publish a FailureTrigger(KIND_COLLISION) carrying CollisionEvidence.
  // `collision_kind` is "self" or "world"; `link_a`/`link_b` name the colliding
  // entities (robot links, or a world obstacle for the world check).
  void publish_collision_failure(const openral_msgs::msg::ActionChunk& chunk,
                                 const char* collision_kind, const std::string& link_a,
                                 const std::string& link_b, int horizon_step, double min_distance);

  // ADR-0030 world phase — ingest bounded world obstacles into a pre-sized
  // buffer (single-threaded executor → no lock needed).
  void on_world_collision(const openral_msgs::msg::WorldCollision::SharedPtr msg);

  // ADR-0030 voxel phase — ingest a dense occupancy grid into a pre-sized buffer.
  void on_world_voxels(const openral_msgs::msg::OccupancyVoxels::SharedPtr msg);

  // ADR-0040 — measured joint-state seed for non-position-mode collision checks.
  // /joint_states feeds q_meas_ (in the action's dof order, mapped by joint
  // name) so a velocity chunk can be reconstructed into the configurations FK
  // can place. Single-threaded executor → direct write, no lock.
  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg);

  // True iff a measurement has landed within `collision_state_deadline_s_` AND
  // every FK-relevant dof has been observed at least once. Fail-closed gate for
  // seed-requiring modes: an incomplete/stale seed must reject, never check a
  // wrong (zero-filled) configuration.
  bool measured_state_fresh() const noexcept;

  // Subscriptions / publishers / service / timer.
  rclcpp::Subscription<openral_msgs::msg::ActionChunk>::SharedPtr candidate_sub_;
  rclcpp::Subscription<openral_msgs::msg::WorldCollision>::SharedPtr world_sub_;
  rclcpp::Subscription<openral_msgs::msg::OccupancyVoxels>::SharedPtr voxel_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr estop_sub_;
  rclcpp_lifecycle::LifecyclePublisher<openral_msgs::msg::ActionChunk>::SharedPtr safe_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Empty>::SharedPtr estop_pub_;
  rclcpp_lifecycle::LifecyclePublisher<openral_msgs::msg::FailureTrigger>::SharedPtr failure_pub_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estop_reset_srv_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;

  // Loaded envelope (populated on_configure).
  EnvelopeIntersection envelope_;
  bool envelope_loaded_{false};

  // ADR-0030 — self-collision model (populated on_configure; disabled by
  // default so manifests without collision geometry behave exactly as before).
  CollisionModel collision_model_;
  CollisionScratch collision_scratch_;
  std::vector<std::string> collision_link_names_;
  bool self_collision_enabled_{false};
  double self_collision_margin_m_{0.0};
  std::size_t collision_required_dof_{0};

  // ADR-0030 world phase — bounded world-obstacle buffer + freshness tracking.
  WorldModel world_model_;
  std::vector<std::string> world_labels_;
  bool world_collision_enabled_{false};
  double world_collision_margin_m_{0.0};
  double world_collision_deadline_s_{0.5};
  std::size_t world_collision_max_primitives_{0};
  bool world_received_{false};
  bool world_overflow_{false};
  rclcpp::Time world_stamp_{};

  // ADR-0030 voxel phase — dense occupancy grid (octomap path). `voxel_grid_`
  // is a view into the pre-sized `voxel_occupancy_` buffer.
  VoxelGrid voxel_grid_;
  std::vector<std::uint8_t> voxel_occupancy_;
  bool world_voxel_enabled_{false};
  double world_voxel_margin_m_{0.0};
  double world_voxel_deadline_s_{0.5};
  std::size_t world_voxel_max_cells_{0};
  bool voxel_received_{false};
  bool voxel_overflow_{false};
  rclcpp::Time voxel_stamp_{};

  // ADR-0040 — measured joint-state seed (Phase 1) + velocity-mode reconstruction
  // (Phase 2). All sized to n_dof at configure; the hot path never allocates.
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  std::vector<std::string> collision_joint_names_;  ///< action-dof-order joint names
  std::unordered_map<std::string, int> joint_name_to_dof_;
  std::vector<double> q_meas_;            ///< latest measured config, dof order
  std::vector<bool> q_meas_seen_;         ///< per-dof: a measurement has landed
  std::vector<int> collision_fk_dofs_;    ///< dof indices FK actually consumes
  std::vector<int> collision_base_dofs_;  ///< mobile-base dofs zeroed for base-relative FK
  std::vector<double> q_check_;           ///< velocity-integration accumulator (no alloc)
  std::vector<double> q_fk_;              ///< per-config FK input (base zeroed; no alloc)
  bool q_meas_received_{false};
  rclcpp::Time q_meas_stamp_{};
  double collision_seed_dt_s_{0.0};         ///< velocity-integration step (s); 0 → reactive only
  double collision_state_deadline_s_{0.2};  ///< max measured-state age for seed-modes

  // ADR-0040 Phase 3 — predictive Cartesian (CARTESIAN_DELTA) look-ahead via the
  // damped-least-squares Jacobian. Reconstructs the per-step joint config the EE
  // deltas drive toward and checks the full capsule boundary at each step (last
  // step always; intermediate steps up to the budget). Reactive measured-config
  // check is the guaranteed floor, so this is purely additive early warning.
  int collision_ee_link_{-1};              ///< EE collision-link index; <0 disables predict
  double collision_predict_lambda_{0.05};  ///< DLS damping (rad/m near singularities)
  double collision_predict_margin_growth_m_{
      0.01};                                    ///< per-step margin inflation (bounds DLS residual)
  std::size_t collision_predict_max_steps_{0};  ///< cap on look-ahead steps; 0 → all rows
  std::vector<double> q_predict_;               ///< predictive-IK accumulator (no alloc)
  std::vector<double> dq_;                      ///< per-step joint increment (no alloc)
  std::vector<std::uint8_t> dof_blocked_;       ///< base dofs excluded from the arm Jacobian

  // Runtime parameters.
  double estop_reset_cooldown_s_{kDefaultEstopResetCooldownSec};

  // Latch + counters.
  bool fault_latch_{false};
  std::chrono::steady_clock::time_point last_estop_at_{};
  std::uint64_t chunks_passed_{0};
  std::uint64_t chunks_dropped_{0};
  std::string last_drop_reason_;
};

}  // namespace openral_safety_kernel

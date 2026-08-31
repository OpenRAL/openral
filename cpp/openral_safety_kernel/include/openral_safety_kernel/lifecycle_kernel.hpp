// SPDX-License-Identifier: Apache-2.0
// SafetyKernelLifecycleNode: the rclcpp_lifecycle::LifecycleNode
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
#include <openral_msgs/msg/safety_status.hpp>
#include <openral_msgs/msg/world_collision.hpp>
#include <openral_msgs/msg/world_state_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/u_int64.hpp>
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

  // ADR-0096 — record the CURRENT safety state and, when it changed,
  // publish it on the latched /openral/safety_status.
  //
  // Transition-gated on the (latched, drop_reason) pair: a fail-closed drop
  // repeats for every chunk while its cause persists (the envelope is still
  // unconfigured, the world model is still stale), and this is called from
  // the chunk callback — republishing per chunk would put a publish on the
  // 30-200 Hz path for no new information. `publish_safety_status_now`
  // refreshes header.stamp at the 1 Hz diagnostics cadence instead, which is
  // what lets a consumer tell a live latch from a dead publisher's leftover
  // durable sample (hazard-log HZ-0096-1).
  //
  // The member message is reused across calls so a transition assigns into
  // already-owned string capacity rather than building a fresh message; the
  // pass-through path returns on the two integer comparisons without
  // touching a string at all.
  void set_safety_status(bool latched, std::uint8_t drop_reason, const char* detail,
                         const std::string& rskill_id, const std::string& trace_id);

  // Stamp and publish `status_msg_` as-is (no transition gate). Used for the
  // activation publish (HZ-0096-1 mitigation 1 — a restarted publisher must
  // overwrite any stale durable value within one activation cycle) and the
  // 1 Hz liveness refresh. No-op while the publisher is deactivated.
  void publish_safety_status_now();

  // Load the self-collision model from ROS parameters (configure
  // time; allocation OK). Returns false with `error` set on a malformed model.
  bool load_collision_model(std::string& error);

  // Publish a FailureTrigger(KIND_COLLISION) carrying CollisionEvidence.
  // `collision_kind` is "self" or "world"; `link_a`/`link_b` name the colliding
  // entities (robot links, or a world obstacle for the world check).
  // `min_distance` MUST be `CollisionHit::min_distance` — the distance of the
  // very pair `link_a`/`link_b` names. The sweep-wide
  // `CollisionHit::sweep_min_distance` belongs to no named pair and never
  // enters the evidence payload (it is logged separately by the caller).
  // `joint_positions` MUST be the configuration forward kinematics was run on
  // for the step being reported (`q_fk_`), so the verdict is re-derivable
  // offline: a predicted step's configuration exists nowhere else, and
  // adjudicating a predictive stop against the measured joints reads geometry
  // the kernel never checked.
  void publish_collision_failure(const openral_msgs::msg::ActionChunk& chunk,
                                 const char* collision_kind, const std::string& link_a,
                                 const std::string& link_b, int horizon_step, double min_distance,
                                 const std::vector<double>& joint_positions);

  // World phase — ingest bounded world obstacles into a pre-sized
  // buffer (single-threaded executor → no lock needed).
  void on_world_collision(const openral_msgs::msg::WorldCollision::SharedPtr msg);

  // Voxel phase — ingest a dense occupancy grid into a pre-sized buffer.
  void on_world_voxels(const openral_msgs::msg::OccupancyVoxels::SharedPtr msg);

  // Attached-payload phase — ingest attached collision objects carried on
  // /openral/world_state_fast into the fixed-capacity attached model. A grasped
  // payload leaves world occupancy and becomes collision-active robot geometry.
  // Malformed / over-capacity / unknown-link attachments fail closed (the next
  // candidate action is dropped until a clean message lands). Single-threaded
  // executor → direct write, no lock.
  void on_world_state(const openral_msgs::msg::WorldStateStamped::SharedPtr msg);

  // Place-phase declaration (ADR-0097 + its 2026-08-14 amendment) — resolve the
  // declaration riding the world state into the payload-scoped approach region
  // the attached-voxel check reduces its margin inside. Every refusal path
  // yields NO region, i.e. exactly the pre-amendment margins: a bad region can
  // only ever make the kernel more permissive, so dropping it is the
  // conservative direction (unlike a malformed support witness, which fails the
  // whole message closed). Called from on_world_state, never on the hot path.
  void ingest_place_declaration(const openral_msgs::msg::WorldStateStamped& msg);

  // Is the ingested declaration still in force at `now`? Retraction, the
  // dispatcher's own timeout backstop, and a stamp from the future all read as
  // dead (HZ-0097-3/4). Re-evaluated per candidate action, so an allowance
  // cannot outlive its declaration between world-state messages.
  bool place_declaration_live() const noexcept;

  // Measured joint-state seed for non-position-mode collision checks.
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
  rclcpp::Subscription<openral_msgs::msg::WorldStateStamped>::SharedPtr world_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr estop_sub_;
  rclcpp_lifecycle::LifecyclePublisher<openral_msgs::msg::ActionChunk>::SharedPtr safe_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Empty>::SharedPtr estop_pub_;
  rclcpp_lifecycle::LifecyclePublisher<openral_msgs::msg::FailureTrigger>::SharedPtr failure_pub_;
  rclcpp_lifecycle::LifecyclePublisher<openral_msgs::msg::SafetyStatus>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr attachment_applied_pub_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estop_reset_srv_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;

  // Loaded envelope (populated on_configure).
  EnvelopeIntersection envelope_;
  bool envelope_loaded_{false};

  // Self-collision model (populated on_configure; disabled by
  // default so manifests without collision geometry behave exactly as before).
  CollisionModel collision_model_;
  CollisionScratch collision_scratch_;
  std::vector<std::string> collision_link_names_;
  bool self_collision_enabled_{false};
  double self_collision_margin_m_{0.0};
  std::size_t collision_required_dof_{0};

  // World phase — bounded world-obstacle buffer + freshness tracking.
  WorldModel world_model_;
  std::vector<std::string> world_labels_;
  bool world_collision_enabled_{false};
  double world_collision_margin_m_{0.0};
  double world_collision_deadline_s_{0.5};
  std::size_t world_collision_max_primitives_{0};
  bool world_received_{false};
  bool world_overflow_{false};
  rclcpp::Time world_stamp_{};

  // Voxel phase — dense occupancy grid (octomap path). `voxel_grid_`
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
  /// Frame the occupancy grid is published in. A place region declared in any
  /// other frame is refused: a region measured in one frame and applied in
  /// another is a relaxation aimed at the wrong volume.
  std::string voxel_frame_id_;

  // Attached-payload phase — grasped objects moved out of world occupancy and
  // re-checked as collision-active robot geometry (ADR-0092). Fixed-capacity
  // preallocated storage sized at configure; the hot path never allocates.
  AttachedModel attached_model_;
  std::vector<std::string> attached_labels_;  ///< per-object id label, capacity = max objects
  bool attached_collision_enabled_{false};
  double attached_collision_margin_m_{0.0};
  double attached_collision_deadline_s_{0.5};
  std::size_t attached_max_objects_{0};
  std::size_t attached_max_primitives_{0};
  std::size_t attached_max_touch_links_{0};
  bool attached_received_{false};
  bool attached_overflow_{false};
  rclcpp::Time attached_stamp_{};
  std::uint64_t attached_revision_{0};
  bool attached_contact_snapshot_pending_{false};
  bool attached_contact_active_{false};
  /// Identity of the support-contact attestation currently armed for one
  /// payload slot. The attachment set is heartbeated, so the kernel re-arms a
  /// witness only when this key changes — otherwise a republished snapshot
  /// would resurrect an exemption that separation had already killed.
  struct SupportWitnessKey {
    std::string object_id;
    std::string support_id;
    std::int64_t stamp_ns{0};
    bool valid{false};
    bool operator==(const SupportWitnessKey& other) const noexcept {
      return valid == other.valid && stamp_ns == other.stamp_ns && object_id == other.object_id &&
             support_id == other.support_id;
    }
  };
  std::uint8_t support_witness_live_{0};  ///< bit i: object i's witness is still live
  std::vector<SupportWitnessKey> support_witness_keys_;
  double support_witness_max_patch_radius_m_{0.0};
  double support_witness_max_penetration_m_{0.0};
  /// Place-phase declaration state (ADR-0097 + its 2026-08-14 amendment).
  /// `place_region_` is the resolved, validated region; the stamp/timeout pair
  /// is the liveness key the kernel re-evaluates itself rather than trusting the
  /// producer to stop publishing (HZ-0097-3 mitigation 2's backstop, which
  /// HZ-0097-4 inherits). `place_declaration_target_` exists only so a stop at a
  /// reduced margin can name the declaration that reduced it (HZ-0097-2
  /// mitigation 1); nothing branches on it.
  PlaceApproachRegion place_region_{};
  std::int64_t place_declaration_stamp_ns_{0};
  double place_declaration_timeout_s_{0.0};
  std::string place_declaration_target_;
  /// Last announced place-region refusal, as (reason token, target). The
  /// attachment set is heartbeated, so a refusal is normally a standing state
  /// rather than an event — the pre-grasp `no_object` case holds for the entire
  /// approach — and re-announcing it per message buries the transitions that do
  /// matter. Refusals are logged on a change of this pair only; the standing
  /// state stays readable on the 1 Hz `/diagnostics` `place_region` key.
  std::string place_region_refusal_reason_;
  std::string place_region_refusal_target_;
  std::vector<std::uint8_t> attached_contact_mask_;
  std::vector<double> attached_contact_distance_;
  std::vector<AttachedObjectInput> attached_ingest_scratch_;  ///< reused across messages

  // Measured joint-state seed (Phase 1) + velocity-mode reconstruction
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

  // Phase 3 — predictive Cartesian (CARTESIAN_DELTA) look-ahead via the
  // damped-least-squares Jacobian. Reconstructs the per-step joint config the EE
  // deltas drive toward and checks the full capsule boundary at each step (last
  // step always; intermediate steps up to the budget). Reactive measured-config
  // check is the guaranteed floor, so this is purely additive early warning.
  int collision_ee_link_{-1};              ///< EE collision-link index; <0 disables predict
  double collision_predict_lambda_{0.05};  ///< DLS damping (rad/m near singularities)
  double collision_predict_margin_growth_m_{
      0.01};  ///< margin growth after first step (bounds accumulated DLS residual)
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
  /// Consecutive advisory refusals (#176) — a payload contact inside its own
  /// declared place region, refused without latching. Reset by any accepted
  /// chunk; at `place_advisory_max_consecutive_` the next one latches like any
  /// other stop, so a robot cannot sit in the band shoving a shelf forever.
  std::uint64_t advisory_refusals_{0};
  std::uint64_t place_advisory_max_consecutive_{0};

  // ADR-0096 — the current /openral/safety_status value, kept as a reusable
  // member so transitions do not build a message from scratch. Default
  // `drop_reason` is 0 (== KIND_TIMEOUT) which is never a state this kernel
  // reports; the activation publish overwrites it with DROP_NONE before any
  // consumer can read it.
  openral_msgs::msg::SafetyStatus status_msg_;
};

}  // namespace openral_safety_kernel

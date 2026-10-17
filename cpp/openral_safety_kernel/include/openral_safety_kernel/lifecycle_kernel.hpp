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

#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <limits>
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

/// Hard caps on the world-voxel freshness parameters, enforced at configure
/// whenever `world_voxel_enabled` (hazard log Entries 033/034). Mirror
/// `openral_core.DeployRuntime`'s caps (`world_voxel_deadline_s <= 2.0`,
/// `world_voxel_data_age_budget_s <= 3.0`, default 1.5) so a node launched
/// outside `openral deploy` cannot run looser than a validated scene;
/// `tests/unit/test_perception_caps_mirror.py` pins both sides equal.
inline constexpr double kMaxWorldVoxelDeadlineMs = 2000.0;
inline constexpr double kMaxWorldVoxelDataAgeBudgetMs = 3000.0;
inline constexpr double kDefaultWorldVoxelDataAgeBudgetMs = 1500.0;

/// Cap on `grasp_region_max_age_s` / `place_region_max_age_s` — how old a
/// producer-measured region's `stamp_ns` may be and still exempt anything. A
/// region is measured from the voxel grid, so it may never be trusted for
/// longer than twice the oldest grid the kernel will check against
/// (`kMaxWorldVoxelDeadlineMs`). `0` (the default) derives the bound as
/// `2 x world_voxel_deadline_ms`; configure refuses anything outside
/// (0, cap] once resolved while the matching allowance is enabled.
inline constexpr double kMaxRegionMeasurementAgeS = 2.0 * kMaxWorldVoxelDeadlineMs / 1000.0;

class SafetyKernelLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
  /// Test seam (#349 review F): when set, called with `true` as the workspace brake
  /// starts deciding on a refused chunk and `false` when it has decided — before any
  /// report, log or publish — so a test can count the decision's allocations (zero).
  /// Unset in production: one relaxed atomic load per refused chunk.
  static inline std::atomic<void (*)(bool)> workspace_brake_probe{nullptr};

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
  /// Accepted chunks republished at a reduced rate by the #188 band. A subset
  /// of `chunks_passed()`, never a subset of `chunks_dropped()`.
  std::uint64_t chunks_scaled() const noexcept { return chunks_scaled_; }
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
  // entities (robot links, or an occupied voxel cell for the voxel check).
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

  // Is a producer-measured region stamped `stamp_ns` young enough to exempt
  // anything at `now`? False when older than `max_age_s`, stamped in the
  // future, or when `max_age_s` is not positive (fails closed). Same clock as
  // every other freshness check here (`this->now()`).
  bool region_measurement_fresh(std::int64_t stamp_ns, double max_age_s) const noexcept;

  // Grasp-phase declaration (ADR-0115 draft, hazard HZ-0115) — resolve the
  // producer-measured grasp declaration riding the world state into the
  // contact-link-scoped region `check_voxel_collision` exempts. Only called
  // with `grasp_allowance_enabled`. Every refusal yields NO region, i.e. the
  // unchanged world-voxel margin. Transition-only logs.
  void ingest_grasp_declaration(const openral_msgs::msg::WorldStateStamped& msg);

  // Issue #418 K1 — ingest the tracker's `tracked_objects` set into the
  // fixed-capacity table `tracked_`. The WHOLE set is refused (every object
  // ineligible, `safety.tracked_objects_rejected reason=<r>` on a change) on a
  // frame mismatch with the grid, a cap breach (box past the grasp-region caps,
  // box geometry, a hull past kMaxTrackedHullVertices), a revision below the
  // last accepted one, or a stamp in the future. Data only: K3 reads it.
  void ingest_tracked_objects(const openral_msgs::msg::WorldStateStamped& msg);
  // Issue #418 K3 — the binding check. For every accepted attachment whose
  // evidence kind is TRACKED_OBJECT, record in `attached_binding_` whether it
  // names a live (confirmed / occluded / bound), non-provisional object of the
  // last accepted set, same id and epoch, whose box meets the hand (`tracked_box_meets_hand`) at
  // the MEASURED configuration. If not, the binding is off the hand: the payload
  // stays checked, but it is never a grasp handover and arms no support
  // witness. WARN (<= 1 Hz) `safety.attachment_binding_off_hand`.
  void check_attachment_bindings(const openral_msgs::msg::WorldStateStamped& msg);
  // Issue #418 K4 — the monotone bound geometry. Rewrites every TRACKED_OBJECT
  // input in `attached_ingest_scratch_` (parallel to `msg.attached_objects`) to
  // the union `bound_` holds for its (object_id, epoch, attach link), in the
  // frame of that binding's first `pose_in_link`. New geometry is unioned in
  // (exact duplicates dropped), never replaces; past kMaxBoundPrimitives it is
  // refused whole, the union kept, WARN <= 1 Hz (`safety.tracked_geometry_cap`).
  // A lower or equal revision never removes anything; one that carries new
  // geometry is still unioned and WARNed (`safety.tracked_geometry_unrevised`).
  // A new epoch starts fresh. Legacy inputs are untouched. Ingest path only.
  void union_bound_geometry(const openral_msgs::msg::WorldStateStamped& msg);
  // Issue #418 K5 — the jaw-open release. NOT monotone (hazard OT-7). For each
  // TRACKED_OBJECT binding at the hand, the measured jaw (`tracked_release_jaw_*`,
  // the one whose column drives the attach or a touch link) is captured at
  // attach as the hold; open past hold + `tracked_release_gap` for
  // `tracked_release_dwell_s` and the kernel freezes the payload in the base
  // frame at its measured pose and opens the (id, epoch) release window: the
  // hand and touch links stay exempt against it until FK separation >
  // `tracked_release_clear_m` or `tracked_release_timeout_s`, after which it is
  // a plain obstacle (no exemption) while the wire carries that binding. The
  // frozen payload is checked against everything else throughout, is never a
  // handover and arms no witness; a handover latched on it retires. Called after
  // `check_attachment_bindings`, on every world-state message. No-op when no jaw
  // joint is configured (the default: today's semantics). Configure refuses it
  // off `hal_mode: sim`, as it does K2.
  void apply_tracked_release();
  // Issue #418 K2 — record the live grasp declaration's `declaration_key`
  // (active, a target, `timeout_s` in range, not retired). Ingest path only.
  void ingest_tracked_declaration_key(const openral_msgs::msg::WorldStateStamped& msg);
  // Issue #418 K2 — fill `tracked_grasp_` with the tracked objects eligible NOW:
  // role TARGET, the live declaration's key, confirmed / occluded, (id, epoch)
  // not retired, confirmed within `grasp_region_max_age_s`, the set's frame still
  // the grid's, and the kernel's cover check on the current grid (re-run once
  // per new grid or set). Candidate path: no allocation of its own (its one
  // transition log aside). Returns the count.
  std::size_t fill_tracked_grasp();
  // Issue #418 K7 — the kernel's cover check on the current grid, re-run once
  // per new grid or set (shared by K2 and K7). Returns whether the set's frame
  // is still the grid's.
  bool refresh_tracked_cover();
  // The tracked set's stream (world_state_fast) is received, accepted and within
  // the attached-collision deadline.
  bool tracked_stream_fresh();
  // Issue #418 K7 — fill `workspace_explained_` with the tracked objects that
  // explain workspace cells NOW (any role; confirmed / occluded; (id, epoch) not
  // retired; confirmed within `grasp_region_max_age_s`; cover-checked; the set's
  // frame the grid's; the stream fresh) and the declared static cells of the
  // current grid. Candidate path: no allocation of its own (its one transition
  // log aside). Returns the object count.
  std::size_t fill_workspace_explained();
  // Issue #418 K7 — re-mark `static_cells_` from `static_points_` when the grid's
  // lattice (pose, resolution, size) changed. World-voxel callback only.
  void refresh_static_cells();
  // Issue #418 K7 — the octomap bridge's points-file format (`x y z` per line,
  // blank lines and `#` comments skipped). False, `out` empty, on any malformed
  // line, an unreadable file or no points. Configure-time.
  static bool load_static_points(const std::string& path, std::vector<Vec3>& out,
                                 std::string& error);

  // Is the ingested grasp region in force at `now`, before the handover rule?
  // Dead on: no region, retracted (no region is ingested), `timeout_s`
  // lapsed, a future stamp, or a world-state stream older than
  // `attached_collision_deadline` (stale is "no exemption", never a drop by
  // itself). Re-evaluated per candidate action.
  bool grasp_declaration_live() const noexcept;

  // Retire the current grasp declaration for good: the region is dropped and
  // the declaration's identity joins the retired set so its heartbeat cannot
  // re-arm it. Only a new declaration (new target or stamp) can arm again.
  // Logs `safety.grasp_region_dropped reason=<reason>` when a region was armed,
  // else `safety.grasp_declaration_retired` (an armed identity failing its last check).
  // Allocation-free: it runs on the candidate path (`handover_exit`).
  void retire_grasp_declaration(const char* reason);
  /// True while a region is live, or the current declaration armed one before.
  bool grasp_identity_armed() const noexcept;
  // Is a payload attached at `attach_link` on the declaring gripper's chain —
  // a link in `mask` or a non-root ancestor of one? Only such an attachment can
  // be a grasp handover (or retire the declaration as the wrong object).
  bool attachment_on_grasp_chain(int attach_link,
                                 const std::bitset<kMaxGraspMaskLinks>& mask) const noexcept;

  // Manipulation-workspace contact (issue #349, draft; Safety-WG review
  // pending). Resolve the producer-confirmed `workspace_region` riding the
  // grasp declaration into `workspace_region_`. Only called with
  // `workspace_contact_enabled`, after `ingest_grasp_declaration`. Every refusal
  // yields NO workspace, and so does a declaration whose grasp part that ingest
  // refused (a producer error such as a non-allowlisted link: the same message's
  // workspace is not trusted either), and so does one retired because its grasp
  // claim was wrong (`handover_object_mismatch`, `latch_mismatch`). A grasp identity
  // retired by a normal handover exit or release does not end the
  // workspace: carry and place happen after the handover (#349 p12_mix04), so it
  // lives to the declaration's own end — retraction, `timeout_s`, a stale
  // confirmation.
  void ingest_workspace_contact(const openral_msgs::msg::WorldStateStamped& msg);
  // Drop the ingested workspace (no-op when none): logs
  // `safety.workspace_region_dropped reason=<reason>` and the declaration's
  // `safety.workspace_contact_summary` once, then resets the tallies.
  void drop_workspace(const char* reason);
  // Is the ingested workspace in force at `now`? Dead on: no workspace, the
  // declaration's `timeout_s` lapsed or stamped in the future (a retired grasp
  // identity does not end it), the world-state stream older than `attached_collision_deadline`,
  // or the confirmation older than `grasp_region_max_age_s`.
  bool workspace_live() const noexcept;
  // Hand speed bound (m/s) of the configuration FK'd into `collision_scratch_`,
  // relative to `prev` (the hand links' poses at the previous row; updated to
  // this row's): max over the workspace links of `link_displacement_bound` /
  // `workspace_control_period_s_`.
  double workspace_hand_speed(std::vector<Transform>& prev) noexcept;
  // Hand lead (m) of the configuration FK'd into `collision_scratch_` over the
  // measured configuration (`workspace_meas_pose_`): max over the workspace
  // links of `link_displacement_bound`, NOT divided by the period. On a
  // joint-PD arm commanded-minus-measured is what the drives push with.
  double workspace_hand_lead() const noexcept;
  /// Hand link `link`'s reach for the SPEED bound: its geometry reach, or a payload
  /// it carries (`attached_object_reach`) when that reaches further — a carried
  /// box's corners strike at their own speed. The lead (push) keeps the hand's own
  /// reach: the drives' push is set by the joint error, which a longer lever only
  /// magnifies on paper (#349 p3_mix02: 95.6 mm with the payload's reach).
  double workspace_reach(std::size_t link) const noexcept;
  // Forget every previously accepted command (row 0 falls back to measured).
  void clear_workspace_commands() noexcept;
  // Is joint `d`'s previously accepted command still row 0's previous row at
  // `now_ns`? Recorded, not cleared since, its last row due (acceptance +
  // (horizon-1) periods) at or before now, and at most
  // `kWorkspaceCommandFreshPeriods` periods ago.
  bool workspace_command_fresh(std::size_t d, std::int64_t now_ns) const noexcept;
  // Copy the workspace links' current FK poses into `out`.
  void workspace_snapshot(std::vector<Transform>& out) const noexcept;

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
  // True iff every FK-relevant dof has a finite measured velocity from a
  // /joint_states message at most `collision_state_deadline_s_` old at
  // `now_ns`. A message naming a joint without a velocity entry forgets it.
  bool measured_velocity_fresh(std::int64_t now_ns) const noexcept;

  // Subscriptions / publishers / service / timer.
  rclcpp::Subscription<openral_msgs::msg::ActionChunk>::SharedPtr candidate_sub_;
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
  /// `world_voxel_data_age_budget_ms` in seconds; in (0, 3] s whenever the
  /// world check is enabled (configure refuses anything else).
  double world_voxel_data_age_budget_s_{kDefaultWorldVoxelDataAgeBudgetMs / 1000.0};
  /// The grid's `source_stamp` (capture of the newest cloud in it), when set.
  bool voxel_source_known_{false};
  rclcpp::Time voxel_source_stamp_{};
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
  /// Issue #418 K1: the last accepted tracked-object set (`tracked_n_ == 0`
  /// when none was published or the last one was refused). Fixed capacity.
  struct TrackedEntry {
    std::string object_id;
    std::uint32_t epoch{0};
    bool live{false};       ///< status confirmed / occluded / bound
    GraspTargetRegion box;  ///< the containing box, validated as a grasp region
    // Issue #418 K2 eligibility inputs, recorded at ingest.
    bool target{false};       ///< role == "target"
    bool exemptible{false};   ///< status confirmed / occluded (never bound / split / …)
    bool provisional{false};  ///< K2 exemption geometry only, never a K3 binding
    std::string declaration_key;
    std::size_t id_hash{0};          ///< std::hash of object_id (the K6 retirement key)
    std::int64_t confirmed_ns{0};    ///< last_confirmed_stamp_ns (source clock)
    std::uint32_t visible_cells{0};  ///< the tracker's expected-cell count
    bool cover_ok{false};            ///< the kernel's own cover check on the current grid
    /// Issue #418 K7: measured support top less one cell (grid frame z); NaN = none.
    double support_z{std::numeric_limits<double>::quiet_NaN()};
    /// Issue #418 OT-10 amendment: the clearing floor (support top + one cell, base z); NaN = none.
    double floor_z{std::numeric_limits<double>::quiet_NaN()};
  };
  std::array<TrackedEntry, kMaxTrackedObjects> tracked_{};
  std::size_t tracked_n_{0};
  bool tracked_have_revision_{false};
  std::uint64_t tracked_revision_{0};
  std::string tracked_refusal_reason_;  ///< last announced refusal (logged on change)
  std::string tracked_set_frame_;       ///< frame of the last accepted set (K2 re-checks it)
  /// Issue #418 K2, the grasp exemption by tracked-object id
  /// (`tracked_grasp_exemption_enabled`, sim-only, requires
  /// `grasp_allowance_enabled`). While on, `tracked_grasp_` replaces the legacy
  /// region as the exemption source of the world-voxel check, every candidate.
  bool tracked_grasp_enabled_{false};
  TrackedGraspTable tracked_grasp_{};
  bool tracked_cover_dirty_{true};  ///< a new grid or set: re-run the cover check once
  /// The live grasp declaration's key, `<target_id>@<stamp_ns>` (what the tracker
  /// stamps on the tracks it grounds or re-keys for that declaration).
  bool tracked_decl_active_{false};
  std::string tracked_decl_key_;
  std::int64_t tracked_decl_stamp_ns_{0};
  double tracked_decl_timeout_s_{0.0};
  std::size_t tracked_grasp_logged_n_{static_cast<std::size_t>(-1)};
  /// Issue #418 K7, explained cells for workspace contact
  /// (`workspace_explained_cells_enabled`, requires `workspace_contact_enabled`).
  /// While on, `workspace_explained_` narrows the #349 workspace exemption to
  /// explained cells, every candidate.
  bool workspace_explained_enabled_{false};
  WorkspaceExplanation workspace_explained_{};
  std::vector<Vec3> static_points_;         ///< `static_points_file`, base frame
  std::vector<std::uint8_t> static_cells_;  ///< pre-sized to world_voxel_max_cells
  std::size_t static_cells_marked_{0};      ///< cells marked for `static_cells_for_`
  VoxelGrid static_cells_for_{};            ///< the lattice `static_cells_` was marked on
  bool static_cells_for_valid_{false};
  std::size_t workspace_explained_logged_n_{static_cast<std::size_t>(-1)};
  /// Issue #418 (hazard OT-10 amendment, `payload_own_cells_enabled`, sim-only, requires K2):
  /// a bound payload's measured / grown cells cannot stop that payload. `payload_own_` is
  /// refilled per candidate from the tracked set and the per-attachment bindings.
  bool payload_own_enabled_{false};
  PayloadOwnCells payload_own_{};
  std::size_t payload_own_logged_n_{static_cast<std::size_t>(-1)};
  std::chrono::steady_clock::time_point payload_own_last_warn_{};
  std::size_t fill_payload_own();
  /// Issue #418 K3: per accepted attachment, its binding. `off_hand` is the
  /// refusal reason (nullptr = a legacy payload or a binding at the hand).
  struct AttachedBinding {
    bool tracked{false};
    std::uint32_t epoch{0};
    const char* off_hand{nullptr};
    /// Issue #418 OT-10 amendment (`payload_own_cells_enabled`): this binding's own track, valid
    /// right now, with a floor. `own_track` indexes `tracked_`.
    bool own_armed{false};
    int own_track{-1};
    double own_floor_z{std::numeric_limits<double>::quiet_NaN()};
  };
  std::vector<AttachedBinding> attached_binding_;  ///< capacity = max objects
  CollisionScratch binding_scratch_;  ///< measured-configuration FK, sized at configure
  std::chrono::steady_clock::time_point binding_last_warn_{};
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
  /// Backing store for `place_region_.geometry` (ADR-0098). Sized once at
  /// configure to `kMaxPlaceTargetPrimitives` and never resized, so the view the
  /// region hands the hot path can never dangle behind a reallocation; the
  /// scratch is the by-name decode of the same list, reused across messages so
  /// the ingest allocates nothing per world-state beat.
  std::vector<AttachedPrimitive> place_geometry_;
  std::vector<AttachedPrimitiveInput> place_geometry_scratch_;
  std::int64_t place_declaration_stamp_ns_{0};
  /// Measurement age bounds for the producer regions (`*_region_max_age_s`,
  /// resolved at configure) and the ingested place region's own `stamp_ns`.
  double place_region_max_age_s_{0.0};
  double grasp_region_max_age_s_{0.0};
  std::int64_t place_region_stamp_ns_{0};
  std::int64_t grasp_region_stamp_ns_{0};
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
  /// Grasp-phase declaration state (ADR-0115 draft). Off unless
  /// `grasp_allowance_enabled`; `grasp_allowlist_` is the launch-derived set of
  /// contact links (`grasp_contact_links`, resolved at configure — an unknown
  /// name fails configure). `grasp_region_` is the validated region whose mask
  /// is the declaration's `contact_links` (all of which must be allowlisted).
  bool grasp_allowance_enabled_{false};
  std::bitset<kMaxGraspMaskLinks> grasp_allowlist_{};
  GraspTargetRegion grasp_region_{};
  std::int64_t grasp_declaration_stamp_ns_{0};
  double grasp_declaration_timeout_s_{0.0};
  std::string grasp_declaration_target_;
  /// Identities (target, stamp) of every retired declaration, up to
  /// `kGraspRetiredCapacity` (oldest evicted, logged once per activation). The
  /// world state is heartbeated, so without this a retired exemption would
  /// re-arm on the next beat; a retired identity never arms again — for every
  /// pick of a multi-pick goal, not only the last one.
  RetiredGraspSet grasp_retired_{};
  /// Issue #418 K6: tracked identities (object_id hash, epoch) whose handover
  /// latched and then retired. A retired (id, epoch) is never a handover again;
  /// the tracker re-grounds a released object under a new epoch, which may arm.
  /// Coexists with `grasp_retired_`, which still keys every declaration.
  RetiredGraspSet tracked_retired_{};
  bool grasp_retired_overflow_logged_{false};
  /// Region latched at the handover edge, keyed by (target, stamp) like the
  /// retirement memory. Later snapshots of that declaration cannot move or
  /// resize it, so a producer re-measuring the carried payload at its live pose
  /// cannot extend the exemption by dragging the box along with it; nor change
  /// its contact links or held object (`grasp_latched_object_`), which retires it.
  GraspTargetRegion grasp_latched_region_{};
  std::string grasp_latched_object_;
  std::uint32_t grasp_latched_epoch_{0};  ///< the latched payload's epoch (#418; 0 = legacy)
  bool grasp_latched_tracked_{false};     ///< the latched payload is a TRACKED_OBJECT binding
  /// Identity (target, stamp) of the declaration that last armed a region; see
  /// `grasp_identity_armed`.
  std::string grasp_armed_target_;
  std::int64_t grasp_armed_stamp_ns_{0};
  std::string grasp_latched_target_;
  std::int64_t grasp_latched_stamp_ns_{0};
  bool grasp_latched_{false};
  bool grasp_latched_moved_warned_{false};
  /// Last announced refusal (reason, target): refusals are logged on a change
  /// only; the standing state is on the 1 Hz `/diagnostics` `grasp_region` key.
  std::string grasp_region_refusal_reason_;
  std::string grasp_region_refusal_target_;
  /// Manipulation-workspace contact (#349, draft). Off unless
  /// `workspace_contact_enabled` (which requires `grasp_allowance_enabled`).
  /// `workspace_links_` is the launch-derived hand-link mask
  /// (`workspace_contact_links`, resolved at configure); `workspace_link_index_`
  /// lists the same links, `workspace_link_reach_` their geometry reach (per
  /// link index). The control period is the chunk row spacing the hand speed
  /// is measured over (`workspace_contact_control_period_s`, from the
  /// manifest's `action_spec.control_freq_hz`).
  bool workspace_contact_enabled_{false};
  std::bitset<kMaxGraspMaskLinks> workspace_links_{};
  std::vector<int> workspace_link_index_;
  std::vector<double> workspace_link_reach_;
  double workspace_max_speed_mps_{0.0};
  double workspace_max_lead_m_{0.0};  ///< `workspace_contact_max_lead_m` (push bound)
  double workspace_control_period_s_{0.0};
  // Lead allowance of the row last checked: `step` periods at the speed bound (0 for
  // row 0, the measured state, and every row while `ws_hand_contact_`). Disclosure only.
  double ws_lead_allow_m_{0.0};
  // The measured hand (or its payload) already touches something: a blocked hand does
  // not travel with the command, so no row gets the per-row allowance (#349 review A).
  bool ws_hand_contact_{false};
  // The checked row's hand speed came from the measured velocity (true) or from the
  // commanded row-to-row motion (false): the larger of the two governs (#349 review B).
  bool ws_speed_measured_row_{false};
  std::chrono::steady_clock::time_point attached_fail_closed_last_warn_{};
  // The row that set the last creep scale and its lead (disclosure only).
  double ws_creep_lead_m_{0.0};
  int ws_creep_row_{-1};
  WorkspaceContactRegion workspace_region_{};
  std::int64_t workspace_region_stamp_ns_{0};
  std::int64_t workspace_declaration_stamp_ns_{0};
  double workspace_declaration_timeout_s_{0.0};
  std::string workspace_target_;
  std::size_t workspace_target_hash_{0};
  std::string workspace_refusal_reason_;  ///< last announced refusal (logged on change)
  /// The last grasp declaration ingested was refused (a producer error, not a stale
  /// measurement): its workspace is refused with it (#349 review E).
  bool grasp_declaration_refused_{false};
  /// The last declaration (target hash, stamp) retired because its grasp claim was
  /// wrong — `handover_object_mismatch`, `latch_mismatch` — not a normal release or
  /// handover exit: its workspace is disarmed and does not re-arm on its heartbeat
  /// (#349 review E). A new declaration arms afresh.
  bool grasp_wrong_claim_{false};
  std::size_t grasp_wrong_claim_hash_{0};
  std::int64_t grasp_wrong_claim_stamp_ns_{0};
  /// Disclosure tallies for the armed declaration (summary on drop).
  std::uint64_t workspace_pairs_total_{0};
  double workspace_min_distance_total_{std::numeric_limits<double>::infinity()};
  double workspace_speed_max_seen_{0.0};
  double workspace_lead_max_seen_{0.0};
  /// Per-link last WARN time (≤ 1 Hz per link), sized to the model's links.
  std::vector<std::chrono::steady_clock::time_point> workspace_last_warn_;
  /// Hand-link poses at the previous checked row (plain / tick-partner rows),
  /// sized to the model's links at configure; never allocated on the hot path.
  std::vector<Transform> workspace_prev_plain_;
  std::vector<Transform> workspace_prev_partner_;
  std::vector<Transform> workspace_creep_prev_;  ///< the brake creep's previous scaled row
  std::vector<double> q_workspace_;              ///< row 0's previous configuration (n_dof)
  /// The jaw columns (`workspace_contact_jaw_joints`) and the row they are reset
  /// to measured in, for the lead:
  /// the lead bounds the ARM's push, a closing jaw's squeeze is the gripper's own
  /// effort limit (#349 p_mix02: a closure stroke read as an 80 mm lead).
  std::vector<int> workspace_jaw_cols_;
  std::vector<double> q_lead_;
  /// Hand-link poses at the measured configuration (the lead's reference).
  std::vector<Transform> workspace_meas_pose_;
  /// The last ACCEPTED JOINT_POSITION command per joint (its chunk's last row),
  /// when its last row is due (node clock, ns), and whether one is held. Sized
  /// n_dof at configure. `workspace_cmd_drops_seen_` is `chunks_dropped_` as of
  /// the last candidate: any refusal since forgets every command.
  std::vector<double> workspace_cmd_q_;
  std::vector<std::int64_t> workspace_cmd_due_ns_;
  std::vector<std::uint8_t> workspace_cmd_set_;
  std::uint64_t workspace_cmd_drops_seen_{0};
  /// The candidate's hand speed source: true = measured (FK of q_meas_ +
  /// dq_meas_ * period vs q_meas_, row-independent, in
  /// `workspace_measured_speed_mps_`); false = the commanded-speed fallback.
  bool workspace_speed_measured_{false};
  /// #349 workspace brake (`workspace_contact_brake_enabled`, needs the workspace
  /// rule): a JOINT_POSITION chunk whose ONLY reasons to stop are the workspace's
  /// measured hand speed and/or lead is replaced by a hold at the measured pose
  /// (jaws keep their command) instead of an E-stop. `ws_dry_run_` runs the
  /// checks without reporting; `ws_quiet_` also skips the workspace disclosure;
  /// `ws_speed_override_` reads the workspace hand speed and lead as 0.
  bool workspace_brake_enabled_{false};
  bool ws_dry_run_{false};
  bool ws_quiet_{false};
  bool ws_speed_override_{false};
  std::uint64_t workspace_brakes_{0};
  /// Holds answering a COMMAND the relaxed check still refuses (the measured pose
  /// passing): consecutive count and its cap (`workspace_contact_hold_max_consecutive`,
  /// 0 = such commands stop as before). Counted per commanded joint (length n_dof,
  /// sized at configure): a chunk passing as-is resets only the joints it commands,
  /// so another arm's slot passing never resets this arm's run (#349 review).
  std::vector<std::uint64_t> workspace_command_holds_;
  std::uint64_t workspace_hold_max_consecutive_{0};
  /// A named geometry of a trip, formatted only when reported or logged: `tag` 'l'
  /// robot link, 'a' attached payload, 'v' voxel cell, 'p' place-target primitive,
  /// 'f' place force gate.
  struct WsName {
    char tag{'l'};
    int idx{-1};
  };
  /// A trip recorded during a brake dry run without allocating (#349 review F).
  struct WsTrip {
    bool set{false};
    const char* kind{""};
    WsName a{};
    WsName b{};
    int step{0};
    CollisionHit hit{};
    double hand_speed_mps{0.0};
    double hand_lead_m{0.0};
    bool speed_measured{false};
  };
  /// The as-is dry run's first trip (and `q_fk_` at it, sized n_dof at configure):
  /// the verdict reported when the brake does not answer it — never re-checked.
  WsTrip ws_first_trip_{};
  std::vector<double> ws_trip_q_;
  /// The first trip of the speed/lead-relaxed pass: why a brake was refused.
  WsTrip ws_refused_trip_{};
  std::chrono::steady_clock::time_point workspace_brake_last_warn_{};
  double workspace_measured_speed_mps_{std::numeric_limits<double>::infinity()};
  bool workspace_speed_max_measured_{false};  ///< source of `workspace_speed_max_seen_`
  std::vector<std::uint8_t> attached_contact_mask_;
  std::vector<double> attached_contact_distance_;
  std::vector<AttachedObjectInput> attached_ingest_scratch_;  ///< reused across messages
  /// Issue #418 K4: the held bound geometry per TRACKED_OBJECT binding, fixed
  /// capacity. An entry outlives a message that omits it (a dropped-and-re-added
  /// record keeps its union); a slot not on the last wire is reused when needed.
  static constexpr std::size_t kMaxBoundPrimitives = 16;
  struct BoundPayload {
    bool used{false};
    bool seen{false};  ///< on the last accepted wire
    std::string object_id;
    std::string attach_link;
    std::uint32_t epoch{0};
    std::uint64_t revision{0};
    Transform frame{};  ///< the first revision's pose_in_link: the union's object frame
    std::array<AttachedPrimitiveInput, kMaxBoundPrimitives> prims{};
    std::size_t n{0};
    // K5, per binding.
    enum class Release : std::uint8_t { kHeld, kWindow, kClosed };
    Release release{Release::kHeld};
    int jaw{-1};  ///< index into release_jaw_cols_; -1 = not resolved yet
    bool hold_set{false};
    double hold{0.0};  ///< measured jaw at attach
    std::int64_t open_since_ns{-1};
    std::int64_t window_opened_ns{0};
    Transform frozen{};  ///< base-frame pose at release
    int hand_link{-1};   ///< the attach link that held it
  };
  std::array<BoundPayload, kMaxTrackedObjects> bound_{};
  std::chrono::steady_clock::time_point bound_cap_last_warn_{};
  std::vector<int> bound_slot_;  ///< per accepted attachment, its `bound_` slot (-1 = none)
  /// Issue #418 K5 parameters (`tracked_release_*`), resolved at configure.
  std::vector<int> release_jaw_cols_;
  std::vector<double> release_jaw_sign_;
  double release_gap_{0.0};
  double release_dwell_s_{0.0};
  double release_clear_m_{0.0};
  double release_timeout_s_{0.0};

  // Measured joint-state seed (Phase 1) + velocity-mode reconstruction
  // (Phase 2). All sized to n_dof at configure; the hot path never allocates.
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  std::vector<std::string> collision_joint_names_;  ///< action-dof-order joint names
  std::unordered_map<std::string, int> joint_name_to_dof_;
  std::vector<double> q_meas_;     ///< latest measured config, dof order
  std::vector<bool> q_meas_seen_;  ///< per-dof: a measurement has landed
  /// Latest measured joint velocity per dof (dof order), whether the last
  /// message naming that joint carried a finite one, and when it landed (ns).
  std::vector<double> dq_meas_;
  std::vector<std::uint8_t> dq_meas_seen_;
  std::vector<std::int64_t> dq_meas_stamp_ns_;
  std::vector<int> collision_fk_dofs_;    ///< dof indices FK actually consumes
  std::vector<int> collision_base_dofs_;  ///< mobile-base dofs zeroed for base-relative FK
  std::vector<double> q_check_;           ///< velocity-integration accumulator (no alloc)
  std::vector<double> q_fk_;              ///< per-config FK input (base zeroed; no alloc)
  // ADR-0102 slot rows (a JOINT_POSITION chunk whose joint_names leave an FK
  // dof uncommanded): those joints are checked at their measured pose, and
  // again at the target an accepted slot of the SAME tick committed for them.
  // Sized to n_dof at configure; the hot path never allocates.
  std::vector<std::uint8_t> slot_commanded_;   ///< per-dof: this chunk commands it
  std::vector<double> tick_target_;            ///< per-dof: same-tick accepted slot target
  std::vector<std::uint8_t> tick_target_set_;  ///< per-dof: tick_target_ holds a value
  std::uint64_t tick_target_session_{0};       ///< (session, tick) tick_target_ belongs to
  std::uint32_t tick_target_tick_{0};
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
  std::uint64_t chunks_scaled_{0};
  std::string last_drop_reason_;
  /// Consecutive advisory refusals (#176) — a payload contact inside its own
  /// declared place region, refused without latching. Reset by any accepted
  /// chunk; at `place_advisory_max_consecutive_` the next one latches like any
  /// other stop, so a robot cannot sit in the band shoving a shelf forever.
  std::uint64_t advisory_refusals_{0};
  std::uint64_t place_advisory_max_consecutive_{0};

  /// Distance-graded velocity scaling (issue #188 — "Path A").
  ///
  /// 2026-08-26 five-round battery measured the cost of accept/drop/latch at
  /// a fixed margin: 11 of 15 stops were inside what 25 mm voxel
  /// quantisation alone explains, each mission-ending. Slowing along the
  /// policy's own path is the response the evidence supports (PACS,
  /// arXiv:2511.06385: 0.70 unfiltered / 0.04 under reactive projection /
  /// 0.72 under chunk-level graded braking).
  ///
  /// collision_scale_proximity_m_ == 0.0 disables the mechanism, restoring
  /// today's behaviour exactly — same rollback shape as
  /// place_advisory_max_consecutive: 0. Default: WG-gated enforcement
  /// surface, the A/B battery is what turns it on.
  double collision_scale_proximity_m_{0.0};
  double collision_scale_k_{0.0};
  double collision_scale_min_{1.0};
  /// Last scale actually logged. The accept path runs at chunk rate, so the
  /// `safety.collision_scaled` line is transition-gated on this rather than
  /// emitted per chunk; the 1 Hz `/diagnostics` counter carries the continuous
  /// signal.
  double last_logged_scale_{1.0};
  /// Reused scale target, so a scaled republish does not allocate after the
  /// first one (the vector keeps its capacity). Untouched while scale == 1.
  openral_msgs::msg::ActionChunk scaled_chunk_;

  /// Velocity scale for a chunk whose nearest checked pair cleared its own
  /// gate margin by `slack_m`. 1.0 outside the band; floored at
  /// `collision_scale_min_` so the band slows the robot but cannot stall it.
  [[nodiscard]] double velocity_scale_for(double slack_m) const noexcept;

  // ADR-0096 — the current /openral/safety_status value, kept as a reusable
  // member so transitions do not build a message from scratch. Default
  // `drop_reason` is 0 (== KIND_TIMEOUT) which is never a state this kernel
  // reports; the activation publish overwrites it with DROP_NONE before any
  // consumer can read it.
  openral_msgs::msg::SafetyStatus status_msg_;
};

}  // namespace openral_safety_kernel

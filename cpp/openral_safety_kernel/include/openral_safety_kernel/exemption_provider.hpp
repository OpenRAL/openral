// Copyright 2026 OpenRAL Contributors
// SPDX-License-Identifier: Apache-2.0
//
// The exemption-provider seam (ADR-0121). A provider is a pluginlib plugin the
// kernel loads at configure time (parameter `exemption_provider`, empty = none).
// It may only RELAX a verdict the open kernel is about to reach, and only inside
// bounds the kernel itself enforces:
//
//  * the kernel asks only after every open check and every open exemption
//    (ADR-0115, ADR-0092) has run, and only about a pair that would otherwise
//    trip;
//  * only on a link in the kernel's relaxable-link mask (the manifest's hand
//    links, `exemption_provider_links`), ANDed with every answer;
//  * self and payload pairs only when every robot link of the pair is in the mask;
//  * never for a joint/speed/torque limit, the
//    workspace envelope, a lead bound or a watchdog;
//  * sim only: the kernel refuses to load any provider off `hal_mode: sim`;
//  * fail closed: no provider, a failed load, an API-version mismatch or a
//    `false` from `initialize` all leave the kernel exactly as with no provider.
//
// Every hook on the per-chunk path is `noexcept`, must not allocate, and is
// timed against the kernel's own per-chunk budget. Nothing a hook returns is
// trusted beyond "tolerate this one pair": the kernel keeps the accounting and
// the disclosure (`CollisionHit::provider_pairs`).

#ifndef OPENRAL_SAFETY_KERNEL__EXEMPTION_PROVIDER_HPP_
#define OPENRAL_SAFETY_KERNEL__EXEMPTION_PROVIDER_HPP_

#include "openral_safety_kernel/collision.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <rclcpp/logger.hpp>
#include <rclcpp/node_interfaces/node_parameters_interface.hpp>

namespace openral_safety_kernel {

/// Bumped on any change to this header's layout or hook semantics. A provider
/// built against another version is refused at load (fail closed).
inline constexpr std::uint32_t kExemptionProviderApiVersion = 3;

/// A provider's answer for one pair. Anything but `kTolerate` keeps the open
/// kernel's verdict.
enum class ProviderVerdict : std::uint8_t {
  kRefuse = 0,
  kTolerate = 1,
};

/// Read-only snapshot of one tracked object as the kernel accepted it (after
/// the open K1 ingest; never a K6-retired identity).
struct TrackedObjectView {
  std::size_t id_hash{0};  ///< std::hash of object_id (the K6 key)
  std::uint32_t epoch{0};
  bool live{false};                     ///< confirmed / occluded / bound
  bool released{false};                 ///< the tracker let go of it
  bool target{false};                   ///< role == "target"
  bool contact{false};                  ///< role target or destination
  bool exemptible{false};               ///< confirmed / occluded
  bool provisional{false};              ///< exemption geometry only, never payload
  bool cover_ok{false};                 ///< the kernel's OWN cover check on this grid
  std::uint32_t visible_cells{0};       ///< the tracker's expected visible cells
  std::int64_t confirmed_ns{0};         ///< last confirmation, source clock
  double support_z{0.0};                ///< NaN = none
  double floor_z{0.0};                  ///< NaN = none
  std::size_t declaration_key_hash{0};  ///< std::hash of the track's declaration_key (API v2)
  GraspTargetRegion box{};              ///< the validated containing box
};

/// The live grasp declaration as the kernel's OWN ingest validated it (API v2). The refusal-type
/// checks (frames, ages, backstop, retirement, refusal, wrong claim) stay in the open kernel; a
/// provider reads only their verdicts.
struct DeclarationView {
  bool live{false};            ///< the kernel's ADR-0115 verdict (`grasp_declaration_live`)
  bool refused{false};         ///< the kernel refused the declaration
  bool wrong_claim{false};     ///< the kernel matched a wrong claim to this identity
  bool tracked_active{false};  ///< the K2 identity checks passed (active, target, not retired)
  std::size_t key_hash{0};     ///< std::hash of "<target_id>@<stamp_ns>" (tracked_active only)
  std::int64_t stamp_ns{0};
  double timeout_s{0.0};
  const GraspTargetRegion* grasp_region{nullptr};    ///< kernel-ingested; null when not live
  const WorkspaceContactRegion* workspace{nullptr};  ///< kernel-ingested; null when not live
  /// The robot's hands (per-hand finger link sets and TCP samples), built by the kernel from
  /// `grasp_contact_links` at configure; boxes empty. Null without `grasp_allowance_enabled`.
  const TrackedGraspTable* hands{nullptr};
};

/// Why the kernel reset its per-chunk state (API v2). A provider that keeps command memory forgets
/// it on every reason.
enum class ResetReason : std::uint8_t {
  kOther = 0,
  kConfigure,
  kActivate,
  kCleanup,
  kRefusal,
  kLatch,
  kExternalEstop,
  kEstopReset,
  kDeclarationChanged,
};

/// A JOINT_POSITION chunk the kernel accepted (API v2): its last row, which joints it commanded,
/// its length and when it was accepted (it plays out over `horizon - 1` control periods).
struct CandidateOutcome {
  const double* q_last_row{nullptr};       ///< n_dof
  const std::uint8_t* commanded{nullptr};  ///< n_dof, 1 = commanded; null = every joint
  std::size_t n_dof{0};
  int horizon{0};
  std::int64_t accepted_ns{0};  ///< kernel clock
};

/// Everything a provider may read for one candidate chunk. All pointers are
/// kernel-owned and valid only for the duration of the call that receives the
/// view (or until the next `begin_candidate`). `measured` is the kernel's FK at
/// the same measured state its own checks use (base dofs zeroed).
struct CandidateView {
  const CollisionModel* model{nullptr};
  const VoxelGrid* grid{nullptr};             ///< null = no fresh grid this chunk
  const CollisionScratch* measured{nullptr};  ///< FK at the measured state; null = stale
  const double* q_measured{nullptr};          ///< n_dof, null = stale
  const double* dq_measured{nullptr};         ///< n_dof, null = no fresh velocity
  std::size_t n_dof{0};
  const TrackedObjectView* tracked{nullptr};  ///< tracked set after K1
  std::size_t n_tracked{0};
  const std::uint8_t* relaxable_links{nullptr};  ///< per link index; the kernel's mask
  std::int64_t now_ns{0};                        ///< kernel clock (ROS time)
  bool position_mode{false};                     ///< JOINT_POSITION chunk
  double control_period_s{0.0};
  DeclarationView declaration{};      ///< API v2
  std::int64_t region_max_age_ns{0};  ///< `grasp_region_max_age_s` (API v2)
  bool tracked_frame_ok{false};       ///< the tracked set and the grid share a frame (API v2)
  bool tracked_stream_fresh{false};   ///< the kernel's verdict on the tracked set's stream (API v2)
  std::uint64_t grid_seq{0};          ///< bumped on every grid the kernel ingests (API v2)
  double workspace_period_s{0.0};     ///< the hand-speed measurement's control period (API v2)
  /// The declared static colliders' cells on THIS grid (one byte per cell, non-zero = static),
  /// `static_cells_n` = the grid's cell count; null when there are none or the lattice does not
  /// match (API v2).
  const std::uint8_t* static_cells{nullptr};
  std::size_t static_cells_n{0};
};

/// One (robot link, occupied cell) pair the open kernel is about to trip on.
struct VoxelPairQuery {
  int link{-1};                             ///< robot link index (always in the mask)
  std::size_t cell{0};                      ///< occupancy index
  Vec3 cell_center_base{};                  ///< cell centre, robot base frame
  double distance{0.0};                     ///< surface distance the open check computed
  int row{0};                               ///< checked row of the chunk (0 = measured)
  const CollisionScratch* row_fk{nullptr};  ///< FK at that row
  /// The kernel's own #349 measurement of the hand at this row (API v2; +inf = unknown). The lead
  /// is raw, before any per-row allowance. `row` is the chunk step (-1 = the measured pass).
  double hand_speed_mps{std::numeric_limits<double>::infinity()};
  double hand_lead_m{std::numeric_limits<double>::infinity()};
  bool hand_contact{false};  ///< the measured hand (or its payload) already touches
};

/// Two robot links of one self-collision pair about to trip (both in the mask).
struct SelfPairQuery {
  int link_a{-1};
  int link_b{-1};
  double distance{0.0};
  int row{0};
  const CollisionScratch* row_fk{nullptr};
  /// The kernel's own #349 measurement of the hand at this row (API v2; +inf = unknown). The lead
  /// is raw, before any per-row allowance. `row` is the chunk step (-1 = the measured pass).
  double hand_speed_mps{std::numeric_limits<double>::infinity()};
  double hand_lead_m{std::numeric_limits<double>::infinity()};
  bool hand_contact{false};  ///< the measured hand (or its payload) already touches
};

/// A carried payload about to trip against an occupied cell (`payload_voxel_pair`) or against a
/// robot link (`payload_self_pair`). Its attach link (and, for a self pair, `other_link`) are in
/// the mask. A frozen payload is offered only to `payload_self_pair`, only in its release window,
/// on the hand that let it go (`release_hand_link`, masked).
struct PayloadPairQuery {
  std::size_t object{0};                         ///< index into the kernel's AttachedModel
  const AttachedModel* attached_model{nullptr};  ///< its primitives and hulls
  const AttachedObject* attached{nullptr};
  Transform object_xf{};  ///< the payload's pose, robot base frame
  int attach_link{-1};
  /// payload_self_pair only: a FROZEN payload (attach_link -1) in its K5 release window, offered
  /// on the hand that let it go; -1 otherwise.
  int release_hand_link{-1};
  int other_link{-1};       ///< payload_self_pair only
  std::size_t cell{0};      ///< payload_voxel_pair only
  Vec3 cell_center_base{};  ///< payload_voxel_pair only
  Vec3 cell_center_grid{};  ///< payload_voxel_pair only, grid axes
  double distance{0.0};
  int row{0};
  const CollisionScratch* row_fk{nullptr};
  /// The kernel's own #349 measurement of the hand at this row (API v2; +inf = unknown). The lead
  /// is raw, before any per-row allowance. `row` is the chunk step (-1 = the measured pass).
  double hand_speed_mps{std::numeric_limits<double>::infinity()};
  double hand_lead_m{std::numeric_limits<double>::infinity()};
  bool hand_contact{false};  ///< the measured hand (or its payload) already touches
};

/// One tracked object whose kernel cover check fell short this grid, while the kernel's OWN check
/// passed for it within `max_age_ns` (API v3). Holding it keeps `cover_ok` until that pass ages
/// out; the kernel never re-stamps a held pass.
struct CoverHoldQuery {
  TrackedObjectView object{};
  std::int64_t last_pass_ns{0};  ///< the kernel's own last passing check, kernel clock
  std::int64_t now_ns{0};
  std::int64_t max_age_ns{0};
  const CollisionScratch* measured{nullptr};  ///< FK at the fresh measured state
  /// The kernel-ingested workspace (its hand-link mask and box), null when not live.
  const WorkspaceContactRegion* workspace{nullptr};
  double resolution{0.0};  ///< this grid's cell size
};

/// A bound (carried) tracked payload at a fresh measured state (API v3). Releasing it
/// freezes it in the base frame where it is; the kernel then keeps its own release window (the
/// hand and touch links exempt against it until `tracked_release_clear_m` clear by FK or
/// `tracked_release_timeout_s`) and closes it, exactly as the in-tree K5 does.
struct ReleaseQuery {
  std::size_t object{0};
  const AttachedModel* attached_model{nullptr};
  const AttachedObject* attached{nullptr};
  std::size_t object_id_hash{0};  ///< std::hash of the binding's object_id
  std::uint32_t epoch{0};
  int attach_link{-1};
  bool on_hand{false};                ///< the kernel's K3 binding check holds it on the hand
  const double* q_measured{nullptr};  ///< n_dof, fresh
  std::size_t n_dof{0};
  std::int64_t now_ns{0};
  const TrackedObjectView* tracked{nullptr};  ///< the tracked set (its `released` flags)
  std::size_t n_tracked{0};
};

/// The #349 workspace brake, requested by a provider (API v3). The brake MECHANISM stays the
/// kernel's: on a refused JOINT_POSITION chunk it re-checks the rows with the hand read as
/// still (speed and lead 0, quoted to the provider's pair queries as such); if that passes it
/// holds the measured pose or creeps toward it, every replacement re-checked by every open check.
/// The provider only enables it and supplies the bounds the creep is scaled by.
struct BrakeConfig {
  bool enabled{false};
  double max_speed_mps{0.0};              ///< finite, > 0
  double max_lead_m{0.0};                 ///< finite, (0, kMaxWorkspaceContactLeadM]
  std::uint64_t hold_max_consecutive{0};  ///< refused commands held in a row before the stop
};

/// The interface. Default implementations refuse everything, so a provider
/// overrides only the hooks it relaxes.
class ExemptionProvider {
public:
  virtual ~ExemptionProvider() = default;

  /// Must return `kExemptionProviderApiVersion` as compiled into the provider.
  virtual std::uint32_t api_version() const noexcept = 0;

  /// Name and version, quoted in every disclosure.
  virtual const char* name() const noexcept = 0;

  /// Configure time, off the hot path; may allocate. `params` lets the provider
  /// declare and read its own parameters (prefix them with its name). Return
  /// false to decline: the kernel then runs with no provider.
  virtual bool initialize(const CollisionModel& model,
                          rclcpp::node_interfaces::NodeParametersInterface::SharedPtr params,
                          const rclcpp::Logger& logger) = 0;

  /// True: the provider REPLACES the open ADR-0115 grasp region as the grasp exemption source
  /// (API v2), as in-tree K2 does today. The kernel then stops applying the region (it only
  /// narrows the open kernel); read once at load and logged.
  virtual bool replaces_grasp_region() const noexcept { return false; }

  /// The workspace brake this provider wants (API v3); disabled by default. Read once at load and
  /// validated by the kernel; needs the hand measurement (`workspace_contact_control_period_s`).
  virtual BrakeConfig brake_config() const noexcept { return {}; }

  /// Per candidate chunk, before any pair query. Preallocated state only.
  virtual void begin_candidate(const CandidateView& view) noexcept { (void)view; }

  /// A JOINT_POSITION chunk was accepted (API v2). A refusal surfaces as
  /// `on_reset(kRefusal)` before the next candidate.
  virtual void end_candidate(const CandidateOutcome& outcome) noexcept { (void)outcome; }

  /// The kernel reset its per-chunk state (API v2).
  virtual void on_reset(ResetReason why) noexcept { (void)why; }

  /// May this robot link touch this occupied cell?
  virtual ProviderVerdict voxel_pair(const VoxelPairQuery& q) noexcept {
    (void)q;
    return ProviderVerdict::kRefuse;
  }

  /// May these two robot links touch? (API v2)
  virtual ProviderVerdict self_pair(const SelfPairQuery& q) noexcept {
    (void)q;
    return ProviderVerdict::kRefuse;
  }

  /// May this carried payload touch this occupied cell? (API v2)
  virtual ProviderVerdict payload_voxel_pair(const PayloadPairQuery& q) noexcept {
    (void)q;
    return ProviderVerdict::kRefuse;
  }

  /// Is this bound payload released now? (API v3; world-state ingest path, not per chunk)
  virtual ProviderVerdict release_payload(const ReleaseQuery& q) noexcept {
    (void)q;
    return ProviderVerdict::kRefuse;
  }

  /// May this object keep a cover the kernel no longer measures? (API v3)
  virtual ProviderVerdict hold_cover(const CoverHoldQuery& q) noexcept {
    (void)q;
    return ProviderVerdict::kRefuse;
  }

  /// May this carried payload touch this robot link? (API v2)
  virtual ProviderVerdict payload_self_pair(const PayloadPairQuery& q) noexcept {
    (void)q;
    return ProviderVerdict::kRefuse;
  }
};

}  // namespace openral_safety_kernel

#endif  // OPENRAL_SAFETY_KERNEL__EXEMPTION_PROVIDER_HPP_

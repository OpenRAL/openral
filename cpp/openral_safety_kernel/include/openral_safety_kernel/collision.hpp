// SPDX-License-Identifier: Apache-2.0
// Allocation-free geometric self-collision core.
//
// Hand-rolled forward kinematics + closed-form capsule-capsule distance, with
// NO external dependency (no Eigen / KDL / Pinocchio) so the safety kernel
// stays small and auditable. All hot-path functions are
// allocation-free: the model is built once at configure time and the FK
// scratch is pre-sized and reused, so they only touch caller-owned storage.
//
// Frames: a capsule's segment runs along its local +Z from -half_length to
// +half_length, swept by `radius` (the MJCF/URDF capsule convention).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace openral_safety_kernel {

/// Plain 3-vector.
struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

/// Rigid transform: row-major 3x3 rotation `r` + translation `t`. POD, stack
/// friendly. Default-constructs to the identity.
struct Transform {
  double r[9]{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  Vec3 t{};
};

/// Joint connecting a link to its parent.
enum class JointKind : std::uint8_t {
  kFixed = 0,
  kRevolute = 1,
  kPrismatic = 2,
};

/// A capsule attached to a link, expressed in that link's frame.
struct Capsule {
  double radius{0.0};
  double half_length{0.0};
  Transform origin{};
};

/// An oriented box (OBB) attached to a link, expressed in that link's frame.
/// The box spans `[-half_extents.k, +half_extents.k]` along each local axis k;
/// `origin` places+orients it in the link frame (same convention as `Capsule`).
/// A box fits a blocky link (a near-cubic housing, e.g. the SO-ARM100/101
/// `base`) far tighter than a capsule, whose circular cross-section must bulge
/// past the block's flat faces and over-report clearance (issue #84).
struct Obb {
  Vec3 half_extents{};
  Transform origin{};
};

/// Flattened kinematic + collision model. Links are topologically ordered so
/// every parent index is < its children's. A link may carry zero, one, or
/// several capsules (real MJCF bodies often have several collision geoms);
/// `capsule_link[c]` names the link capsule `c` is rigidly attached to. Built
/// once at configure time.
struct CollisionModel {
  std::size_t n_links{0};
  std::vector<int> parent;            ///< parent link index; -1 for the root
  std::vector<JointKind> joint_kind;  ///< joint connecting parent -> this link
  std::vector<int> dof_index;         ///< qpos index for revolute/prismatic; -1 if fixed
  std::vector<Transform> origin;      ///< fixed parent-link -> joint transform
  std::vector<Vec3> axis;             ///< joint axis (unit) in the joint frame
  std::vector<int> capsule_link;      ///< link index each capsule attaches to
  std::vector<Capsule> capsules;      ///< parallel to capsule_link
  std::vector<int> box_link;          ///< link index each OBB attaches to
  std::vector<Obb> boxes;             ///< parallel to box_link (blocky links)
  std::vector<std::pair<int, int>> allowed_pairs;  ///< unordered link pairs to skip
};

/// Pre-sized scratch reused across calls; resize `link_world` to `n_links`
/// once at configure time so the hot path never allocates.
struct CollisionScratch {
  std::vector<Transform> link_world;  ///< per-link frame in the base frame
};

/// Bounded set of world obstacles, each a capsule already expressed in the
/// robot base frame (`origin` is the absolute base-frame transform — no link
/// composition). Ingested from perception into a pre-sized buffer.
struct WorldModel {
  std::vector<Capsule> capsules;
};

/// A dense, fixed-capacity 3-D occupancy voxel grid in the robot base frame —
/// the kernel-facing form of a 3-D world map (e.g. an OctoMap lowered by a
/// perception bridge into a bounded local volume). `occupancy` is row-major
/// with x fastest (`idx = x + sx*(y + sy*z)`); a cell is occupied when its
/// value is non-zero. A view: `occupancy` points at a buffer the caller owns.
struct VoxelGrid {
  Vec3 origin{};           ///< base-frame position of voxel (0,0,0)'s min corner
  double resolution{0.0};  ///< voxel edge length (m)
  int sx{0};               ///< grid dimensions
  int sy{0};
  int sz{0};
  const std::uint8_t* occupancy{nullptr};              ///< sx*sy*sz cells, non-zero = occupied
  const std::uint8_t* attached_contact_mask{nullptr};  ///< bit i: object i had attach-time contact
  const double* attached_contact_distance{nullptr};    ///< object-major baseline distance
  std::size_t attached_contact_stride{0};              ///< cells per object in baseline buffer
  double attached_contact_tolerance{0.0};              ///< physical slack on an attested depth (m)
  std::uint8_t support_witness_live{0};                ///< bit i: object i's witness is still live
};

/// One collision check's outcome — and, on a hit, the E-stop evidence.
///
/// `link_a`, `link_b` and `min_distance` always describe **one and the same**
/// geometry pair: on a hit they are the deepest pair that actually tripped the
/// check's gate. They must never be sampled from different pairs — evidence
/// that names one cell and quotes another cell's distance sends downstream
/// diagnosis after a penetration that does not exist (an attached payload's
/// exempt attach-time contact residue reading as if it were the fresh support
/// contact that stopped the robot).
///
/// `sweep_min_distance` is the separate, sweep-wide figure: the minimum
/// surface distance over **every** pair the check touched, including pairs
/// that stayed clear of the margin and pairs the gate deliberately exempted.
/// It is a diagnostic, not the reason for the stop — keep it in its own field
/// and its own log key.
///
/// With no hit there is no pair to describe, so `min_distance` keeps its
/// clearance meaning and equals `sweep_min_distance`. Both are `+inf` when the
/// check compared nothing.
struct CollisionHit {
  bool hit{false};
  int link_a{-1};
  int link_b{-1};
  double min_distance{0.0};        ///< the reported pair's surface distance (clearance if no hit)
  double sweep_min_distance{0.0};  ///< minimum over every checked pair, gated or exempted
};

/// Convex shape of a collision object rigidly attached to a robot link
/// (grasped payload). Numeric values match
/// ``openral_msgs/AttachedCollisionObject.SHAPE_*`` so the ingest path can
/// decode the wire field without translation.
enum class AttachedShapeKind : std::uint8_t {
  kSphere = 1,
  kCapsule = 2,
  kBox = 3,
};

/// One convex collision primitive belonging to an attached object, expressed
/// in the object's frame by `pose_in_object`. A sphere is a capsule with
/// `half_length == 0`. On each checked configuration the kernel composes it
/// through FK: `scratch.link_world[attach_link] · object.pose_in_link ·
/// pose_in_object` into the base frame.
struct AttachedPrimitive {
  AttachedShapeKind kind{AttachedShapeKind::kSphere};
  double radius{0.0};          ///< sphere / capsule radius
  double half_length{0.0};     ///< capsule half-length (0 for a sphere)
  Vec3 half_extents{};         ///< box half-extents
  Transform pose_in_object{};  ///< primitive pose in the owning object's frame
};

/// One collision object rigidly attached to a robot link (a grasped payload).
/// The object keeps a single identity, attach link, touch-link set, and an
/// object pose in the attach-link frame (`pose_in_link`), but may own several
/// primitives. `prim_first` / `prim_count` index a slice of
/// `AttachedModel.primitives` (the object's geometry); `touch_first` /
/// `touch_count` index a slice of `AttachedModel.touch_links` naming the robot
/// links explicitly allowed to contact this object (the attach link itself is
/// always allowed).
///
/// `has_support_witness` and the four `support_*` fields carry World State's
/// support-contact attestation (ADR-0092 D6) — the bounded statement that this
/// payload rests on a named environment surface. The geometry is expressed in
/// the **object's own frame**, not the base frame and not a voxel index, so it
/// stays valid while the mobile base drives and the voxel lattice re-phases
/// underneath the robot. `support_normal` is the support surface's outward unit
/// normal: it points **from the supporting solid toward the payload**, so the
/// support occupies the half-space `(x - support_point)·support_normal <= 0`.
/// No attestation (`has_support_witness == false`) means no exemption.
struct AttachedObject {
  Transform pose_in_link{};             ///< object pose in the attach-link frame
  int attach_link{-1};                  ///< robot link index the object is attached to
  int prim_first{0};                    ///< offset into AttachedModel.primitives
  int prim_count{0};                    ///< number of primitives owned by this object
  int touch_first{0};                   ///< offset into AttachedModel.touch_links
  int touch_count{0};                   ///< number of touch-link entries for this object
  bool has_support_witness{false};      ///< World State attested a support contact
  Vec3 support_point{};                 ///< attested contact point, object frame
  Vec3 support_normal{0.0, 0.0, 1.0};   ///< unit outward support normal, object frame
  double support_patch_radius{0.0};     ///< lateral radius of the supported patch (m)
  double support_max_penetration{0.0};  ///< attested PHYSICAL contact depth bound (m)
};

/// Bounded, fixed-capacity set of attached payloads ingested from the world
/// state. `objects`, `primitives`, and `touch_links` are pre-sized to their
/// configured caps at configure time; the ingest path fills the first
/// `n_objects` / `n_primitives` entries and the hot path never allocates.
/// `n_objects == 0` means nothing is carried.
struct AttachedModel {
  std::size_t n_objects{0};                   ///< active object count (<= objects.size())
  std::size_t n_primitives{0};                ///< active primitive count (<= primitives.size())
  std::vector<AttachedObject> objects;        ///< capacity = max attached objects
  std::vector<AttachedPrimitive> primitives;  ///< flattened, capacity = max primitives
  std::vector<int> touch_links;               ///< flattened touch-link indices, capacity = cap
};

/// Parsed, still-by-name primitive record produced by the ROS ingest callback
/// (one entry per wire `AttachedCollisionPrimitive`). Not on the hot path.
struct AttachedPrimitiveInput {
  AttachedShapeKind kind{AttachedShapeKind::kSphere};
  double radius{0.0};
  double half_length{0.0};
  Vec3 half_extents{};
  Transform pose_in_object{};
};

/// Parsed, still-by-name attached-object record produced by the ROS ingest
/// callback and handed to `ingest_attached_objects` for capacity + link-name
/// validation. Not on the hot path (built once per world-state message).
struct AttachedObjectInput {
  Transform pose_in_link{};              ///< object pose in the attach-link frame
  std::string attach_link;               ///< attach-link name (resolved against link_names)
  std::vector<std::string> touch_links;  ///< touch-link names (resolved against link_names)
  std::vector<AttachedPrimitiveInput> primitives;  ///< one or more owned primitives
  bool has_support_witness{false};                 ///< wire `support_contact_valid`
  Vec3 support_point{};                            ///< attested contact point, object frame
  Vec3 support_normal{};                ///< outward support normal, object frame (normalised)
  double support_patch_radius{0.0};     ///< lateral patch radius (m)
  double support_max_penetration{0.0};  ///< attested physical contact depth bound (m)
};

/// Outcome of an attachment-ingest attempt. Anything other than `kOk` is
/// fail-closed at the call site (the world state is marked invalid and the next
/// candidate action is dropped until a clean message lands).
enum class AttachIngestStatus : std::uint8_t {
  kOk = 0,
  kOverflow = 1,     ///< object/primitive/touch-link cap exceeded
  kUnknownLink = 2,  ///< attach_link or a touch_link not in the robot model
  kMalformed = 3,    ///< bad primitive dimensions or an object with zero primitives
};

/// Build a rigid transform from a translation and fixed-axis XYZ Euler angles
/// (roll about X, pitch about Y, yaw about Z), i.e. R = Rz(yaw)·Ry(pitch)·Rx(roll)
/// — the URDF / ROS `<origin xyz rpy>` convention. Used at configure time to
/// lower manifest origins into the `CollisionModel`; not on the hot path.
Transform transform_from_xyz_rpy(double x, double y, double z, double roll, double pitch,
                                 double yaw) noexcept;

/// Build a rigid transform from a translation and a (x, y, z, w) unit
/// quaternion — the ``geometry_msgs/Pose`` convention used by
/// ``openral_msgs/AttachedCollisionObject.pose_in_link``. A non-unit quaternion
/// is normalised; a zero quaternion degrades to the identity rotation. Used at
/// ingest time (not the hot path) to lower a wire pose into the kernel's
/// `Transform`.
Transform transform_from_translation_quat(double x, double y, double z, double qx, double qy,
                                          double qz, double qw) noexcept;

/// Closest distance between the surfaces of two capsules, given each capsule's
/// frame in a common frame. Negative means interpenetration. Allocation-free.
double capsule_distance(const Transform& a, double a_radius, double a_half_length,
                        const Transform& b, double b_radius, double b_half_length) noexcept;

/// Closest distance between an OBB surface and a capsule surface, each given in
/// a common frame (`box`/`cap` are the primitive origin transforms). Exact for
/// the disjoint case (segment↔box closest distance minus the capsule radius);
/// negative means interpenetration. Allocation-free.
double box_capsule_distance(const Transform& box, const Vec3& half_extents, const Transform& cap,
                            double cap_radius, double cap_half_length) noexcept;

/// Conservative closest distance between two OBB surfaces. Uses the separating-
/// axis theorem: the maximum gap over the 15 candidate axes (6 face normals + 9
/// edge-edge cross products) is a lower bound on the true surface distance, so
/// the kernel never *under*-reports a collision (safety §3 — at least as
/// conservative). Negative means overlap. Allocation-free.
double box_box_distance(const Transform& a, const Vec3& a_half, const Transform& b,
                        const Vec3& b_half) noexcept;

/// Forward kinematics for one joint-position row (`qpos`, length `n_dof`):
/// fills `scratch.link_world[i]` with each link's frame in the base frame.
/// Allocation-free; `scratch.link_world` must already be sized to
/// `model.n_links`.
void forward_kinematics(const CollisionModel& model, const double* qpos, std::size_t n_dof,
                        CollisionScratch& scratch) noexcept;

/// Check every non-allowed capsule pair against a `margin` clearance using the
/// link frames in `scratch`. On a hit the returned pair is the deepest pair
/// within the margin and `min_distance` is that pair's distance;
/// `sweep_min_distance` carries the minimum over every checked pair
/// (`CollisionHit`). Allocation-free.
CollisionHit check_self_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                  double margin) noexcept;

/// Check every robot capsule (FK'd via `scratch`) against every world obstacle
/// in `world` (base-frame capsules) at a `margin` clearance. On a hit,
/// `link_a` is the robot link index and `link_b` is the world obstacle index
/// of the deepest pair within the margin, and `min_distance` is that pair's
/// distance (`CollisionHit`). Allocation-free.
CollisionHit check_world_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const WorldModel& world, double margin) noexcept;

/// Maximum joint count the allocation-free Jacobian step supports (stack scratch
/// is sized to this). Covers every in-tree robot (humanoids ~30 dof) with
/// headroom; a model exceeding it makes `jacobian_dls_step` fail-safe (returns
/// false → caller falls back to the reactive check).
inline constexpr std::size_t kMaxJacobianDof = 64;

/// Damped-least-squares IK step for predictive Cartesian
/// checking. Forward kinematics must already be run for the current
/// configuration (`scratch`). Given the end-effector link index `ee_link` and a
/// desired base-frame EE twist `ee_twist` (6 = [vx,vy,vz, wx,wy,wz], the
/// per-step Cartesian delta), compute the joint increment `dq` (length `n_dof`,
/// zeroed then filled only for the dofs on the kinematic chain root→ee_link):
///     dq = Jᵀ (J Jᵀ + λ²·I)⁻¹ · ee_twist
/// where `J` is the geometric Jacobian of `ee_link` and `lambda` damps motion
/// near singularities. The caller integrates `q ← q + dq`, re-runs FK, and
/// re-checks the capsule boundary; because DLS can *undershoot* the true motion,
/// the caller must inflate the collision margin to bound the residual (the
/// reactive measured-config check remains the guaranteed floor regardless).
/// `dof_blocked` (nullable, length `n_dof`) excludes any dof whose entry is
/// non-zero from the Jacobian — used on a mobile base to keep the EE twist
/// realised by the arm joints only (the base dofs are not driven by the arm's
/// Cartesian command and are zeroed before the collision FK). Pass nullptr for a
/// fixed-base arm.
/// Allocation-free (fixed stack scratch). Returns false and leaves `dq` zeroed
/// if `ee_link` is out of range, `n_dof > kMaxJacobianDof`, or no usable
/// (non-blocked, movable) joint feeds the EE.
bool jacobian_dls_step(const CollisionModel& model, const CollisionScratch& scratch, int ee_link,
                       const double ee_twist[6], double lambda, double* dq, std::size_t n_dof,
                       const std::uint8_t* dof_blocked = nullptr) noexcept;

/// Check every robot capsule (FK'd via `scratch`) against the occupied cells of
/// a dense voxel `grid`. Only the voxels inside each capsule's inflated AABB
/// are tested (bounded), and each occupied voxel is treated conservatively as a
/// sphere of the voxel half-diagonal at the cell centre. On a hit, `link_a` is
/// the robot link index and `link_b` is the linear index of the deepest cell
/// within the margin, and `min_distance` is that cell's distance
/// (`CollisionHit`). Allocation-free.
CollisionHit check_voxel_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const VoxelGrid& grid, double margin) noexcept;

/// Validate + resolve parsed attached-object inputs into the fixed-capacity
/// `out` model. `out.objects` / `out.primitives` / `out.touch_links` must
/// already be sized to their configured caps (`max_objects`, `max_primitives`,
/// `max_touch_links`); this fills the first N entries and sets `out.n_objects`
/// / `out.n_primitives`. Each object's primitives and touch links are appended
/// to the flattened buffers and referenced by slice. Link names are resolved
/// against `link_names` (the robot collision-link name table). Fail-closed:
/// any cap overflow (objects, total primitives, or total touch links), unknown
/// attach/touch link, malformed primitive shape, or an object carrying zero
/// primitives leaves `out.n_objects == 0` and returns the offending status —
/// the caller must then treat the attachment set as unavailable (drop the
/// candidate action). Not on the hot path (called once per world-state
/// message). Allocation-free with respect to `out` (only reads/writes the
/// pre-sized buffers).
AttachIngestStatus ingest_attached_objects(const std::vector<AttachedObjectInput>& inputs,
                                           const std::vector<std::string>& link_names,
                                           std::size_t max_objects, std::size_t max_primitives,
                                           std::size_t max_touch_links,
                                           AttachedModel& out) noexcept;

/// Check every attached payload (FK'd via `scratch` through its attach link)
/// against every world obstacle capsule at a `margin` clearance. On a hit,
/// `link_a` is the attached-object index and `link_b` is the world obstacle
/// index of the deepest pair within the margin, and `min_distance` is that
/// pair's distance (`CollisionHit`). Allocation-free.
CollisionHit check_attached_world_collision(const CollisionModel& model,
                                            const AttachedModel& attached,
                                            const CollisionScratch& scratch,
                                            const WorldModel& world, double margin) noexcept;

/// Check every attached payload against the occupied cells of a dense voxel
/// `grid` (same conservative per-voxel cube treatment as
/// `check_voxel_collision`). On a hit, `link_a` is the attached-object index
/// and `link_b` is the linear index of the deepest cell that actually tripped
/// the check, and `min_distance` is that cell's distance.
///
/// Two — and only two — exemptions can spare a cell, both bounded:
///
/// 1. **Support-contact witness** (ADR-0092 D6): object `i` carries a World
///    State attestation, its bit is live in `grid.support_witness_live`, and
///    the cell satisfies `support_contact_exempts` — inside the attested patch
///    laterally, and no higher above the attested support plane than the voxel
///    cube's own projected half-width plus the attested physical depth plus
///    `grid.attached_contact_tolerance`. This is what lets a ~1 mm physical
///    support contact survive 25 mm voxels: the depth is measured against the
///    attested plane, so the cell-cube inflation is accounted for exactly
///    instead of being absorbed by a widened tolerance.
/// 2. **Embedded attach-time residue**: the payload's own uncleared occupancy
///    left in the map at attach (a cell already at least half a voxel inside
///    the payload when the baseline was snapshotted). This is stale
///    self-occupancy, not support contact, and the witness deliberately does
///    not cover it.
///
/// An exempted cell never supplies the reported identity or distance — it
/// reaches `sweep_min_distance` only, so an exempt cell can never be published
/// as the contact that stopped the robot (`CollisionHit`). Allocation-free.
CollisionHit check_attached_voxel_collision(const CollisionModel& model,
                                            const AttachedModel& attached,
                                            const CollisionScratch& scratch, const VoxelGrid& grid,
                                            double margin) noexcept;

/// Does object `i`'s attested support contact explain occupied cell `center`?
///
/// The predicate is purely geometric and index-free, so it does not decorrelate
/// as the base drives. With the attested plane lifted into the base frame
/// (`p = obj_xf · support_point`, `n = obj_xf.R · support_normal`) and
/// `s = (center - p)·n` the cell centre's height above that plane:
///
/// * **Bounded laterally** — the cell must lie within `support_patch_radius`
///   of the contact point (padded by the voxel cube's circumradius, which is
///   the exact discretisation slop). A new contact against a wall or a fixture
///   elsewhere is outside the patch and still stops the robot.
/// * **Bounded in depth** — `s <= w + support_max_penetration + slack`, where
///   `w = half_resolution · (|n.x| + |n.y| + |n.z|)` is the *exact* half-width
///   of the voxel cube projected on the support normal. A surface cell of a
///   support flush with the attested plane has `|s| <= w` by construction, so
///   this admits the full quantisation envelope and nothing beyond it: solid
///   sitting genuinely higher than the attested support face trips the check,
///   and a payload driving deeper into its support raises `s` at the physical
///   rate (1 mm of sink = 1 mm of `s`) until it trips.
///
/// The caller must have already established that the witness is live; this
/// function does not consult `grid.support_witness_live`. Allocation-free.
bool support_contact_exempts(const AttachedObject& object, const Transform& object_xf,
                             const Vec3& center, double resolution, double slack) noexcept;

/// Refresh the per-object support-contact witness latch against the measured
/// configuration. Bit `i` of `live_mask` survives only while object `i`'s
/// attestation is still doing work: some occupied cell it would exempt is
/// within `margin` of the payload. Once the payload separates from its attested
/// support, the bit is cleared and this function never sets it again — a fresh
/// attestation from World State is the only way to re-arm, so a regrasp or a
/// re-contact after lift is a new violation, not a resurrected exemption.
/// Returns the updated mask. Allocation-free; supports the first eight attached
/// objects (the schema cap).
std::uint8_t update_support_contact_witnesses(const AttachedModel& attached,
                                              const CollisionScratch& scratch,
                                              const VoxelGrid& grid, std::uint8_t live_mask,
                                              double margin) noexcept;

/// Snapshot or refresh the bounded attach-time voxel contacts allowed for each
/// payload. On `snapshot=true`, occupied cells already intersecting object i
/// receive bit i and their minimum surface distance is recorded. On refresh,
/// each bit is cleared permanently once that payload separates from the cell.
/// Only the *embedded* subset of that baseline still exempts anything: the
/// payload's own uncleared occupancy residue. The non-deepening index-keyed
/// allowance this used to grant was removed — voxel indices shift under a
/// driving base while the physical contact persists, so it decorrelated; the
/// support-contact witness replaced it.
/// Allocation-free; supports the first eight attached objects (the schema cap).
bool update_attached_voxel_contacts(const AttachedModel& attached, const CollisionScratch& scratch,
                                    const VoxelGrid& grid, std::uint8_t* contact_mask,
                                    double* contact_distance, std::size_t mask_capacity,
                                    std::size_t distance_capacity, bool snapshot) noexcept;

/// Check every attached payload against the robot's own link geometry
/// (capsules + boxes), skipping each object's attach link and its explicit
/// touch links (a grasped object legitimately contacts the fingers that hold
/// it). On a hit, `link_a` is the attached-object index and `link_b` is the
/// robot link index of the deepest pair within the margin, and `min_distance`
/// is that pair's distance (`CollisionHit`). Allocation-free.
CollisionHit check_attached_self_collision(const CollisionModel& model,
                                           const AttachedModel& attached,
                                           const CollisionScratch& scratch, double margin) noexcept;

}  // namespace openral_safety_kernel

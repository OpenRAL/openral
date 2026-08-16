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

/// Binding absolute ceiling on the declaration-scoped place approach allowance
/// (ADR-0097's 2026-08-14 amendment, Condition 1, as calibrated by its
/// **Second Amendment 2026-08-15**; hazard log HZ-0097-4 mitigation 1 and its
/// "Calibration 2026-08-15" subsection). The allowance is
/// `min(kPlaceApproachAllowanceVoxels × one voxel, this)`, so a coarser map can
/// only ever *shrink* it — never widen it. Both numbers are maintainer-set
/// conditions, not implementation choices: Entry 010's HZ-0095-2 is the
/// precedent the ceiling exists to prevent from recurring (a 25 mm
/// sim-resolution parameter silently becoming 50 mm at real hardware's 5 cm
/// resolution under a purely resolution-relative formula). Changing either
/// needs a new recorded decision.
///
/// The ceiling was `0.025` until 2026-08-15, when the 5-run battery
/// (`spark:~/openral-runs/2026-08-15-baguette-battery/run1`) showed a one-voxel
/// allowance is *structurally* marginal rather than occasionally short: it is
/// sized to absorb exactly the ~one-voxel map-vs-truth error at a placement
/// pose, leaving nothing for the real contact the place witness is earned by
/// making (run 1 read 26.48 mm of penetration at a −2.43 mm ground-truth
/// contact, 1.48 mm past the old cap).
inline constexpr double kMaxPlaceApproachAllowanceM = 0.04;

/// Voxel multiple in the same allowance (ADR-0097's Second Amendment). 37.5 mm
/// at sim's 25 mm grid; at a real 50 mm grid `1.5 × 50 = 75 mm` is capped by
/// `kMaxPlaceApproachAllowanceM` to 40 mm, which is the whole point of having
/// two numbers.
inline constexpr double kPlaceApproachAllowanceVoxels = 1.5;

/// The allowance cap at a given live voxel `resolution` — `min(1.5 × voxel,
/// 4 cm)`, and `0.0` for an unusable resolution. One definition, so the geometry
/// and the `safety.place_region_armed` log line can never quote different
/// numbers for the same map.
inline constexpr double place_approach_allowance_cap(double resolution) noexcept {
  if (!(resolution > 0.0)) {
    return 0.0;
  }
  const double scaled = kPlaceApproachAllowanceVoxels * resolution;
  return scaled < kMaxPlaceApproachAllowanceM ? scaled : kMaxPlaceApproachAllowanceM;
}

/// Sanity bound on one side of a declared place region. A declaration names ONE
/// receptacle, not a room, so a half-extent past this is a producer error and
/// buys no allowance at all (fail-closed toward the unchanged margin).
inline constexpr double kMaxPlaceRegionHalfExtentM = 1.5;

/// Sanity bound on a declared place region's volume (m³) — a 2 m cube. Same
/// fail-closed direction as `kMaxPlaceRegionHalfExtentM`.
inline constexpr double kMaxPlaceRegionVolumeM3 = 8.0;

/// The producer-supplied region of a live place declaration (ADR-0097's
/// 2026-08-14 amendment), lowered into the kernel's own frame convention: an
/// oriented box whose `pose` is expressed in the **robot base frame** — the same
/// frame the occupancy grid is in, which is why the node only accepts a region
/// whose declared `frame_id` matches the grid's.
///
/// While it is valid, payload-vs-world voxel checks for the objects named in
/// `object_mask` run at a margin reduced by
/// `place_approach_allowance_cap(resolution)` against cells whose **centre** lies
/// inside the box. Everything else is untouched: arm-vs-world, payload-vs-world outside the
/// box, the support-contact witness, and the reported evidence. The region is a
/// license to *approach* the contact the declaration already licenses — the hard
/// stop behind the reduced margin is unchanged, so deepening past the allowance
/// still stops (HZ-0097-4 mitigation 3).
///
/// `valid == false` — no declaration, a retracted or expired one, a region the
/// producer never supplied, or one that failed `ingest_place_region` — means no
/// allowance anywhere, i.e. behaviour identical to before the amendment.
struct PlaceApproachRegion {
  bool valid{false};            ///< a live, validated region is in force
  std::uint8_t object_mask{0};  ///< bit i: attached object i is the declaration's payload
  Transform pose{};             ///< region box centre pose, robot base frame
  Vec3 half_extents{};          ///< region box half-extents (m, all > 0)
};

/// Outcome of a place-region ingest attempt (`ingest_place_region`).
///
/// Every value other than `kOk` means exactly one thing to the geometry — *no
/// allowance*, i.e. the unchanged pre-amendment margin — and something quite
/// different to whoever reads the log, which is why the outcome is not a bool.
/// `kNoObject` is the ordinary pre-grasp state: dispatch declares the place
/// phase before the payload is attached, so the declaration resolves to no
/// carried object on every attachment heartbeat until the grasp lands. The
/// remaining values are producer errors in a message that reached the kernel.
/// Collapsing the two into one `reason=bounds` label is what round-8
/// (`spark:~/openral-runs/2026-08-15-round8/`) reported as 672-811 identical
/// bounds warnings per run, none of which described a bound.
enum class PlaceRegionStatus : std::uint8_t {
  kOk = 0,
  kNoObject = 1,        ///< empty object_mask — the declared payload is not carried
  kBadPose = 2,         ///< non-finite region pose
  kBadExtents = 3,      ///< non-finite half-extent
  kDegenerate = 4,      ///< half-extent <= 0: a box with no interior licenses nothing
  kOversize = 5,        ///< a side past kMaxPlaceRegionHalfExtentM
  kOversizeVolume = 6,  ///< a box past kMaxPlaceRegionVolumeM3 — a room, not a receptacle
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
  PlaceApproachRegion place_region{};                  ///< live place declaration's region, if any
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
/// `place_allowance_active` is disclosure, never a decision (CLAUDE.md §1.4):
/// it is true when the reported pair tripped a margin that the declaration-
/// scoped place approach allowance had *reduced*, so an operator reading the
/// evidence knows the stop happened inside a declared region at a reduced
/// margin. `min_distance` stays the pair's true surface distance either way.
struct CollisionHit {
  bool hit{false};
  int link_a{-1};
  int link_b{-1};
  double min_distance{0.0};        ///< the reported pair's surface distance (clearance if no hit)
  double sweep_min_distance{0.0};  ///< minimum over every checked pair, gated or exempted
  bool place_allowance_active{false};  ///< the reported pair's margin was reduced by the allowance
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
/// While a place declaration's region is live for this payload
/// (`grid.place_region`), the *margin* each cell inside that region is gated
/// against is first reduced by `place_approach_allowance` — the bounded license
/// to reach the contact the declaration already permits (ADR-0097's 2026-08-14
/// amendment, capped per its Second Amendment at `min(1.5 × voxel, 4 cm)`). That is a margin
/// change, not an exemption: a cell deeper than the reduced margin still stops the robot, and the
/// reported distance is the cell's true distance.
///
/// Two — and only two — exemptions can spare a cell outright, both bounded:
///
/// 1. **Support-contact witness** (ADR-0092 D6): object `i` carries a World
///    State attestation, its bit is live in `grid.support_witness_live`, and
///    the cell satisfies `support_contact_exempts` — inside the attested patch
///    laterally, and no higher above the attested support plane than the voxel
///    cube's own projected half-width plus the attested physical depth plus
///    `grid.attached_contact_tolerance` plus one voxel of co-planar headroom.
///    This is what lets a ~1 mm physical support contact survive 25 mm voxels:
///    the depth is measured against the attested plane, so the cell-cube
///    inflation is accounted for exactly instead of being absorbed by a widened
///    tolerance.
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

/// Margin reduction the live place declaration grants object `object_index`
/// against an occupied cell centred at `center` — `0.0` whenever it grants none.
///
/// The whole predicate, and every way it fails closed:
///
/// * No valid region, or a grid with no usable resolution → `0.0`.
/// * `object_index` outside the declaration's `object_mask` (or past the
///   eight-object schema cap) → `0.0`. The allowance follows the *declared*
///   payload, never a second object the robot happens to be carrying.
/// * A degenerate or non-finite region box → `0.0`. Validation belongs to
///   `ingest_place_region`; this re-checks because a permissive allowance is
///   never the safe default.
/// * The cell centre outside the region box (an exact point-in-OBB test in the
///   box's own frame) → `0.0`.
///
/// Otherwise the reduction is `place_approach_allowance_cap(grid.resolution)` —
/// `min(1.5 × voxel, 4 cm)`, the amendment's binding Condition 1 as calibrated
/// by its Second Amendment (2026-08-15), computed here against the *live*
/// resolution so it can never desynchronise from the map actually being checked.
/// Allocation-free.
double place_approach_allowance(const VoxelGrid& grid, std::size_t object_index,
                                const Vec3& center) noexcept;

/// Validate a producer-supplied place region and lower it into `out`.
///
/// Fail-closed means *no allowance* here, not a dropped message: unlike a
/// malformed support-contact attestation (which fails the whole attachment
/// closed because a producer over-claiming measured contact is not one to trust
/// with the payload model), a bad region can only ever make the kernel *more*
/// permissive, so refusing it restores exactly the pre-amendment margin. The
/// caller logs the refusal; nothing else changes.
///
/// Rejects a non-finite pose or half-extent, a non-positive half-extent, a
/// half-extent past `kMaxPlaceRegionHalfExtentM`, a box past
/// `kMaxPlaceRegionVolumeM3`, and an empty `object_mask` (a declaration whose
/// payload is not among the carried objects). Returns `PlaceRegionStatus::kOk`
/// iff `out.valid` was set; every other value leaves `out` inert. Not on the
/// hot path (once per world-state message).
PlaceRegionStatus ingest_place_region(const Transform& pose, const Vec3& half_extents,
                                      std::uint8_t object_mask, PlaceApproachRegion& out) noexcept;

/// Stable snake_case token naming `status`, for the `reason=` key of the
/// kernel's place-region log lines. Never null; unknown values read `unknown`.
const char* place_region_status_reason(PlaceRegionStatus status) noexcept;

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
/// * **Bounded in height** — `s <= w + support_max_penetration + slack +
///   resolution`, where `w = half_resolution · (|n.x| + |n.y| + |n.z|)` is the
///   *exact* half-width of the voxel cube projected on the support normal. A
///   surface cell of a support flush with the attested plane has `|s| <= w` by
///   construction; the fourth term is **one voxel of co-planar headroom**
///   (hazard log Entry 012, "Calibration 2026-08-15"), because cells of adjacent
///   co-planar structure — a raised edge, a neighbouring stack on the same
///   surface — sit about one voxel above the attested plane while the payload is
///   in genuine, continuing support contact (round-8 r2: +42.9 mm against a
///   ~15–19 mm envelope). Past *that* bound, solid sitting genuinely higher than
///   the support face still trips the check, and a payload driving deeper into
///   its support raises `s` at the physical rate (1 mm of sink = 1 mm of `s`)
///   until it trips. The lateral patch bound above is untouched by that
///   calibration: it widens height, never reach.
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

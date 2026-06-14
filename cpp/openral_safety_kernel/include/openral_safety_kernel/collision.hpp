// SPDX-License-Identifier: Apache-2.0
// ADR-0030 phase 2 — allocation-free geometric self-collision core.
//
// Hand-rolled forward kinematics + closed-form capsule-capsule distance, with
// NO external dependency (no Eigen / KDL / Pinocchio) so the safety kernel
// stays small and auditable (ADR-0020 ethos). All hot-path functions are
// allocation-free: the model is built once at configure time and the FK
// scratch is pre-sized and reused, so they only touch caller-owned storage.
//
// Frames: a capsule's segment runs along its local +Z from -half_length to
// +half_length, swept by `radius` (the MJCF/URDF capsule convention).

#pragma once

#include <cstddef>
#include <cstdint>
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
  const std::uint8_t* occupancy{nullptr};  ///< sx*sy*sz cells, non-zero = occupied
};

/// First self-collision hit (and the minimum surface distance observed across
/// all checked pairs, even when nothing collided).
struct CollisionHit {
  bool hit{false};
  int link_a{-1};
  int link_b{-1};
  double min_distance{0.0};
};

/// Build a rigid transform from a translation and fixed-axis XYZ Euler angles
/// (roll about X, pitch about Y, yaw about Z), i.e. R = Rz(yaw)·Ry(pitch)·Rx(roll)
/// — the URDF / ROS `<origin xyz rpy>` convention. Used at configure time to
/// lower manifest origins into the `CollisionModel`; not on the hot path.
Transform transform_from_xyz_rpy(double x, double y, double z, double roll, double pitch,
                                 double yaw) noexcept;

/// Closest distance between the surfaces of two capsules, given each capsule's
/// frame in a common frame. Negative means interpenetration. Allocation-free.
double capsule_distance(const Transform& a, double a_radius, double a_half_length,
                        const Transform& b, double b_radius, double b_half_length) noexcept;

/// Forward kinematics for one joint-position row (`qpos`, length `n_dof`):
/// fills `scratch.link_world[i]` with each link's frame in the base frame.
/// Allocation-free; `scratch.link_world` must already be sized to
/// `model.n_links`.
void forward_kinematics(const CollisionModel& model, const double* qpos, std::size_t n_dof,
                        CollisionScratch& scratch) noexcept;

/// Check every non-allowed capsule pair against a `margin` clearance using the
/// link frames in `scratch`. Returns the first hit; `min_distance` always
/// carries the minimum surface distance seen. Allocation-free.
CollisionHit check_self_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                  double margin) noexcept;

/// Check every robot capsule (FK'd via `scratch`) against every world obstacle
/// in `world` (base-frame capsules) at a `margin` clearance. On a hit,
/// `link_a` is the robot link index and `link_b` is the world obstacle index;
/// `min_distance` carries the minimum surface distance seen. Allocation-free.
CollisionHit check_world_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const WorldModel& world, double margin) noexcept;

/// Maximum joint count the allocation-free Jacobian step supports (stack scratch
/// is sized to this). Covers every in-tree robot (humanoids ~30 dof) with
/// headroom; a model exceeding it makes `jacobian_dls_step` fail-safe (returns
/// false → caller falls back to the reactive check).
inline constexpr std::size_t kMaxJacobianDof = 64;

/// ADR-0040 Phase 3 — damped-least-squares IK step for predictive Cartesian
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
/// the robot link index and `link_b` is the linear voxel index. Allocation-free.
CollisionHit check_voxel_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const VoxelGrid& grid, double margin) noexcept;

}  // namespace openral_safety_kernel

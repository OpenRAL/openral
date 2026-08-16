// SPDX-License-Identifier: Apache-2.0
// Attach-time (and ongoing) clearing of a grasped payload's own occupancy.
//
// `openral_msgs/AttachedCollisionObject` states the contract: "The same object
// must be absent from world occupancy while attached." Before the grasp the
// object legitimately *is* world occupancy — its cells were marked by honest
// sensor returns. At the attach transition those cells stop describing the
// world and start describing the robot's own payload, which the kernel already
// re-checks as collision-active attached geometry. Nothing removed them, so
// they stayed in the map and stopped the arm against the thing it was holding.
//
// This is the removal, done where occupancy is owned (this Layer-2 bridge) and
// on the map itself, not as an exemption inside the kernel: every consumer of
// `/openral/world_voxels` sees the payload gone, and the kernel keeps its one
// unconditional rule — an occupied cell is an obstacle.
//
// The clearing is PARTITIONED against the kernel's support-contact witness
// (ADR-0092 D6). A payload resting on a counter shares its bottom cell layer
// with the counter's own top surface, so a clearing that only knows the
// payload's volume takes the counter with it: the surface the witness is
// attested against disappears from the map, the kernel's occupancy-based
// liveness test finds nothing left to keep the witness alive
// (`update_support_contact_witnesses`), and the very next frame in which a
// support cell falls back outside the clearing reach stops the robot against
// the surface it is resting on — unexempted, because the witness is dead and
// only World State can re-arm it. That is the 2026-08-14 failure, reproduced
// 2/2 on baguette+counter and cup+island, with the nearest surviving cell
// measured 21.77 mm from the payload (one circumradius out).
//
// The clearing also distinguishes the ATTACH TRANSITION from steady state.
// The cells that describe the object before the grasp were marked by a sensor
// that saw the real object, while the clearing measures against a fitted
// convex primitive: primitive-fit error plus lattice quantization leaves a thin
// residue of pre-attach silhouette just past one circumradius, which the
// steady-state reach can never explain. On the 2026-08-14 round-5 baguette run
// that residue was one cell — `voxel_91633`, 22.13 mm from the payload's
// primitive surface at resolution 0.025, i.e. 0.48 mm outside the 21.65 mm
// reach — and it E-stopped the arm against the thing it was carrying. So a
// payload clears with `attach_sweep_padding_m` of extra reach (one voxel by
// default) for as long as it is still ON that silhouette, and with the
// steady-state reach unchanged afterwards.
//
// "Still on it" is a POSITION, not a frame count. The grid is re-rasterized
// from the octree every tick, so clearing the residue once removes it from one
// published grid and the octree hands it straight back — nothing retires an
// occluded cell. The field E-stop fired 2.78 s (~28 grids at 10 Hz) after the
// attach sweep, with the payload still where it was grasped. So the widened
// reach stays open until the payload has translated more than the sweep padding
// away from its attach pose (`AttachSweepLedger`), at which point every cell
// the padding covered is either inside the steady reach or beyond the payload
// entirely. It is deliberately not spent on occupancy the payload has moved
// away from, which is evidence about the world and stops the robot as it should.
//
// So the two mechanisms divide the cells between them, and the division is
// exhaustive: within the payload's reach a cell is either CLEARED by this
// bridge or EXEMPTED by the kernel's witness, never neither, never both.
// `support_patch_withholds` is this side of that line — the same geometry the
// kernel's `support_contact_exempts` uses (including its one voxel of co-planar
// headroom), with zero slack, so what this bridge keeps is a subset of what the
// kernel exempts and no withheld cell can stop the robot. Everything else around the payload clears
// exactly as before: withholding only ever puts occupancy BACK into the published map, which is
// also what Nav2 and SLAM need, because those cells are the counter.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/occupancy_voxels.hpp>

namespace openral_octomap_bridge {

/// One attached-payload collision primitive, already placed in the grid frame.
///
/// The same convex shapes the kernel ingests from
/// `openral_msgs/AttachedCollisionPrimitive`, with the same conventions: a
/// capsule's central segment lies on the primitive frame's local +Z, and a
/// box's `half_extents` are along its own local axes.
struct PayloadPrimitive {
  /// `AttachedCollisionPrimitive::SHAPE_SPHERE` / `_CAPSULE` / `_BOX`.
  std::uint8_t shape_type{0};
  /// Primitive origin expressed in the occupancy grid's frame.
  tf2::Transform pose{tf2::Transform::getIdentity()};
  double radius{0.0};                        ///< sphere / capsule radius (m)
  double half_length{0.0};                   ///< capsule central-segment HALF length (m)
  tf2::Vector3 half_extents{0.0, 0.0, 0.0};  ///< box half extents (m)
};

/// One attested support-contact patch, already placed in the grid frame.
///
/// The wire form (`openral_msgs/SupportContactWitness`, carried on
/// `AttachedCollisionObject.support_contact`) states the geometry in the
/// attached **object's** frame; this is that geometry lifted into the grid
/// frame by the same live pose the payload primitives are placed with, which is
/// where the cells it protects live. `normal` points **from the supporting
/// solid toward the payload**, so the support occupies the half-space
/// `(x - point)·normal <= 0` — the same convention the kernel's
/// `AttachedObject::support_normal` uses.
struct SupportPatch {
  tf2::Vector3 point{0.0, 0.0, 0.0};   ///< attested contact point, grid frame
  tf2::Vector3 normal{0.0, 0.0, 1.0};  ///< unit outward support normal, grid frame
  double patch_radius{0.0};            ///< lateral radius of the supported patch (m, > 0)
  double max_penetration{0.0};         ///< attested PHYSICAL contact depth bound (m, >= 0)
};

/// Place one wire object's primitives — and its support-contact attestation, if
/// it carries one — into the grid frame, appending them to `out` / `out_patches`.
///
/// `grid_from_link` maps a point in the object's attach-link frame into the
/// grid frame (i.e. `lookupTransform(base_frame, object.attach_link)`); each
/// primitive lands at `grid_from_link · pose_in_link · pose_in_object`, the
/// same composition the kernel uses to place the payload it checks. The witness
/// is per-object, not per-primitive, so its point and normal land at
/// `grid_from_link · pose_in_link` (the object frame) and one object appends at
/// most one patch. `support_contact_valid == false` — the honest default for
/// any producer that cannot measure support contact — appends no patch, and the
/// object's cells then clear exactly as they did before the witness existed.
///
/// Fail-closed: an object carrying no primitives, an unrecognised shape tag,
/// too few shape dimensions for the tag, a non-finite or negative dimension, a
/// degenerate (zero) pose quaternion, or a malformed attestation (non-finite
/// contact point, degenerate normal, non-positive patch radius, negative
/// penetration — the kernel's own `ingest_attached_objects` rules) returns
/// `false` and appends **nothing** to either output — the caller must then
/// clear nothing at all. Refusing to clear only ever leaves occupancy in the
/// map, which is the conservative direction; a producer that cannot describe
/// its own support contact is one whose payload volume is not trusted either,
/// exactly as the kernel treats it.
bool place_attached_object(const openral_msgs::msg::AttachedCollisionObject& object,
                           const tf2::Transform& grid_from_link, std::vector<PayloadPrimitive>& out,
                           std::vector<SupportPatch>& out_patches);

/// Does an attested support patch claim the cell centred at `center`, and
/// therefore forbid the clearing from removing it?
///
/// The predicate is the kernel's `support_contact_exempts` with `slack = 0`,
/// evaluated on the same lifted plane: with `s = (center - point)·normal` the
/// cell centre's height above the attested support face, the cell is withheld
/// when it is inside the patch laterally (padded by the cell cube's
/// circumradius, `resolution·√3/2`) **and**
/// `s <= w + max_penetration + resolution`, where
/// `w = half_resolution·(|n.x| + |n.y| + |n.z|)` is the exact half-width of the
/// cell cube projected on the support normal and the third term is the kernel's
/// one voxel of co-planar headroom (hazard log Entry 012, "Calibration
/// 2026-08-15"). That term is mirrored here **in lockstep** with the kernel, by
/// obligation of the same hazard entry: `withheld ⊆ exempt` is true because the
/// two are the same inequality with a strictly tighter bound on this side, and a
/// height term present in one and absent from the other would break it by
/// construction.
///
/// Dropping the kernel's slack term is what makes the partition safe rather
/// than merely symmetric: the kernel adds `attached_contact_tolerance` (1 mm of
/// physical FK/pose slack) to the same bound, so every cell this withholds is a
/// cell the kernel exempts **for the object that attested it**, and no cell kept
/// by this function can be the one that stops the robot against that object.
/// The reverse inclusion is deliberately not claimed — the kernel may exempt a
/// cell this bridge still clears, and a cleared cell stops nothing. Nor does the
/// containment survive two scope conditions the caller must know about: the
/// kernel's exemption is per-object while `clear_attached_payload_cells`
/// withholds per message (see its docs), and the kernel's lifecycle node caps
/// the attestation it will accept (`support_witness_max_patch_radius_m`,
/// `support_witness_max_penetration_m`) where this predicate takes the wire
/// values as given. Both are stated, with their tests, in the package README.
///
/// The withheld region is the support HALF-SPACE — the slab above the plane
/// plus everything below it — never the patch cylinder. At 25 mm cells the slab
/// reaches at most 21.65 mm (the pad, largest for a cube-diagonal normal) plus
/// the attested depth plus one voxel above the plane — 56.65 mm for the widest
/// attestation the kernel would accept — which is what keeps the payload-side
/// residue further above a support surface in the clearing's hands.
///
/// A cell BELOW the plane is withheld without a lower bound: that is the body
/// of the counter, real furniture that the map is supposed to carry and that
/// the payload's volume merely happens to pass within a circumradius of.
bool support_patch_withholds(const SupportPatch& patch, const tf2::Vector3& center,
                             double resolution) noexcept;

/// Clear from `grid` every occupied cell the attached payload's own volume
/// explains **except** those an attested support patch claims, in place.
/// Returns the number of cells cleared.
///
/// A cell is cleared when its centre lies no further from a primitive's surface
/// than the cell cube's circumradius (`resolution·√3/2`) plus `padding_m`, and
/// no patch in `patches` withholds it (`support_patch_withholds`) — every
/// object's attestation guards every object's clearing, which is one step wider
/// than the kernel's per-object exemption and is the one place a withheld cell
/// can still stop the robot (README, "Two scope conditions"). Passing an
/// empty `patches` is the un-partitioned behaviour and clears the support
/// surface out from under a resting payload — see the header comment. Because
/// withholding can only ever *skip* a clear, the cells this removes are always
/// a subset of the cells it would remove with no patches at all: the partition
/// puts occupancy back into the map and never takes any away.
///
/// The circumradius is the exact discretisation slop: every cell the payload's
/// volume actually intersects has its centre within it, so this clears all of
/// them, at the price of also clearing a cell the payload merely comes within
/// one circumradius of — bounded by a single cell layer and unavoidable on a
/// discretised map. It is the same bound `support_contact_exempts` uses in the
/// kernel. `padding_m` (default 0 at the call site) buys pose uncertainty on a
/// real robot and does remove cells the payload cannot explain: it is
/// protection given up, so keep it at 0 unless a measured pose error demands
/// otherwise. The one caller that passes more is the attach transition
/// (`AttachSweepLedger` + `attach_transition_padding`), which spends it on one
/// grid, on the stale pre-attach silhouette.
///
/// Only cells inside each primitive's inflated AABB are visited, so the cost is
/// the payload's volume, not the grid's. Clearing never *marks*: the function
/// only ever writes 0. A grid whose `occupancy` length disagrees with its
/// dimensions, or whose resolution is not positive, is left untouched (0
/// returned).
std::size_t clear_attached_payload_cells(openral_msgs::msg::OccupancyVoxels& grid,
                                         const std::vector<PayloadPrimitive>& primitives,
                                         const std::vector<SupportPatch>& patches,
                                         double padding_m);

/// One attached object's identity for the attach-transition sweep.
///
/// `object_id` is the wire identity; `attachment_revision` is
/// `WorldStateStamped.attachment_revision`, the producer's own counter, bumped
/// once per atomic attachment-set change (`openral_msgs/AttachmentState`) and
/// never per frame. The pair is therefore an object REVISION: it is stable for
/// as long as a payload is carried under one attachment record and changes the
/// moment the producer publishes a new one — a grasp, a release, or a
/// place-phase arm/disarm.
struct AttachedObjectRevision {
  std::string object_id;
  std::uint64_t attachment_revision{0};
};

/// One object's identity and where its payload is, on one published grid.
///
/// `payload_position` must be stated in a frame the stale silhouette does NOT
/// move in — the OctoMap's own frame, not the base frame. The residue is world
/// -fixed occupancy; what the window measures is the payload moving away from
/// it, and on a mobile base a payload that is motionless in `base_link` may be
/// travelling through the map at walking pace.
struct AttachSweepObservation {
  AttachedObjectRevision key;
  tf2::Vector3 payload_position{0.0, 0.0, 0.0};
};

/// The open attach-transition windows: which payloads are still close enough to
/// where they were grasped that the residue of their pre-attach silhouette is
/// still within the widened reach.
///
/// This is the bridge's ONLY memory. It is not a latch on cells — nothing about
/// which cells were cleared is remembered, and no cell is ever held out of a
/// later grid — so the per-frame, derive-everything-from-the-wire property the
/// rest of the clearing has is unchanged. What it buys is the one distinction
/// the steady-state reach cannot make on its own: stale pre-attach silhouette
/// versus occupancy that describes the world.
///
/// The window is POSITION-based, not frame-count-based, and that is not a
/// refinement — it is the difference between fixing the field case and not.
/// The grid is re-rasterized from the octree on every tick, so a one-shot sweep
/// removes the residue from exactly one published grid and the octree re-supplies
/// it on the next: an occluded cell gets no clearing ray, ever. On the 2026-08-14
/// round-5 baguette run the E-stop fired 2.78 s — ~28 published grids at 10 Hz —
/// after the attach, with the payload still on its stale silhouette. So the
/// widened reach stays open until the payload has actually LEFT the residue.
///
/// The closing test is displacement > `window_m` (the sweep padding itself),
/// because that is exactly when the widened reach stops doing work the steady
/// reach cannot: a cell that sat between the steady and padded reach at the
/// attach pose is, once the payload has moved by more than the padding, either
/// inside the steady reach (the payload moved toward it — cleared anyway) or
/// outside the padded reach (the payload moved away — no longer the payload's
/// silhouette to clear). Closing LATCHES: a payload that wanders back does not
/// re-open a window, because by then the map around it is evidence gathered
/// while it was elsewhere.
class AttachSweepLedger {
public:
  /// Which of `present` are still inside their attach-transition window, in
  /// order — and record this grid.
  ///
  /// Calling this IS the commit, which is what makes the answer and the record
  /// impossible to disagree. An object absent from `present` is forgotten, so a
  /// detach — and the next grasp — restarts the window; an object whose
  /// revision the producer has bumped is a new key and starts a new window; a
  /// key seen for the first time is inside its window by construction (its
  /// anchor is the position observed now), which is the attach transition
  /// itself. Not calling it at all — a frame that clears NOTHING (stale
  /// attachment state, missing TF, a payload the bridge will not place) —
  /// leaves every window exactly as it was, so the padded sweep stays owed
  /// rather than being consumed by a frame that swept nothing.
  std::vector<std::uint8_t> sweep(const std::vector<AttachSweepObservation>& present,
                                  double window_m);

  /// How many windows the last swept grid carried (open and closed alike).
  std::size_t size() const noexcept;

private:
  struct Window {
    AttachedObjectRevision key;
    /// Where the payload was when this revision first appeared.
    tf2::Vector3 anchor{0.0, 0.0, 0.0};
    bool closed{false};
  };

  std::vector<Window> windows_;
};

/// The clearing padding one object's cells get on ONE published grid.
///
/// `steady_padding_m` (the `attached_clear_padding_m` parameter, 0 by default)
/// once the object's attach-transition window has closed;
/// `steady_padding_m + attach_sweep_padding_m` while it is open. The attach reach is
/// therefore never TIGHTER than the steady-state reach, and a non-finite or
/// negative `attach_sweep_padding_m` contributes nothing rather than shrinking
/// it — a parameter cannot be used to narrow the clearing below what the
/// payload's own volume explains.
double attach_transition_padding(double steady_padding_m, double attach_sweep_padding_m) noexcept;

}  // namespace openral_octomap_bridge

// SPDX-License-Identifier: Apache-2.0
// Clearing by identity (issue #418): the gate, the hull test, the exclusions and the
// octree cell selection, on real IDL messages and a real octomap::OcTree.

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <octomap/OcTree.h>

#include "openral_octomap_bridge/identity_clearing.hpp"

namespace bridge = openral_octomap_bridge;
using openral_msgs::msg::AttachedCollisionObject;
using openral_msgs::msg::AttachedCollisionPrimitive;
using openral_msgs::msg::TrackedObject;
using openral_msgs::msg::WorldStateStamped;

namespace {

constexpr double kRes = 0.015;

TrackedObject track(const std::string& id, const std::string& status, double x, double z,
                    double half) {
  TrackedObject t;
  t.object_id = id;
  t.epoch = 2;
  t.status = status;
  t.box.frame_id = "base_link";
  t.box.pose.position.x = x;
  t.box.pose.position.z = z;
  t.box.pose.orientation.w = 1.0;
  t.box.half_extents.x = t.box.half_extents.y = t.box.half_extents.z = half;
  t.floor_z = 0.045;
  if (status == "bound") {
    t.bound_hand = "gripper";
    t.attach_link = "hand";
  }
  return t;
}

AttachedCollisionObject attachment(const std::string& evidence, double half) {
  AttachedCollisionObject o;
  o.object_id = "trk:1";
  o.epoch = 2;
  o.attach_link = "hand";
  o.evidence_kind = evidence;
  o.pose_in_link.orientation.w = 1.0;
  AttachedCollisionPrimitive box;
  box.shape_type = AttachedCollisionPrimitive::SHAPE_BOX;
  box.shape_dimensions = {half, half, half};
  box.pose_in_object.orientation.w = 1.0;
  o.primitives.push_back(box);
  return o;
}

WorldStateStamped bound_state(const std::string& evidence = bridge::kTrackedObjectEvidence) {
  WorldStateStamped s;
  s.tracked_objects_valid = true;
  s.tracked_objects.frame_id = "base_link";
  s.tracked_objects.objects.push_back(track("trk:1", "bound", 0.15, 0.08, 0.03));
  s.attached_objects.push_back(attachment(evidence, 0.03));
  return s;
}

const TrackedObject* gate(const WorldStateStamped& s) {
  return bridge::identity_clearing_track(s.attached_objects.front(), s, "base_link");
}

TEST(IdentityGate, AnUnambiguousBoundTrackedObjectIsClearedByIdentity) {
  const auto s = bound_state();
  ASSERT_NE(gate(s), nullptr);
  EXPECT_EQ(gate(s)->object_id, "trk:1");
}

TEST(IdentityGate, APinchFallbackGetsNone) {
  EXPECT_EQ(gate(bound_state(bridge::kPinchFallbackEvidence)), nullptr);
}

TEST(IdentityGate, ALegacyEvidenceKindGetsNone) {
  EXPECT_EQ(gate(bound_state("grasp_target_region")), nullptr);
}

TEST(IdentityGate, AnAmbiguousBindingGetsNone) {
  auto two_tracks = bound_state();
  two_tracks.tracked_objects.objects.push_back(track("trk:2", "bound", 0.16, 0.08, 0.03));
  EXPECT_EQ(gate(two_tracks), nullptr) << "a second object bound to the same hand";
  auto two_payloads = bound_state();
  auto second = attachment(bridge::kTrackedObjectEvidence, 0.03);
  second.object_id = "trk:2";
  two_payloads.attached_objects.push_back(second);
  EXPECT_EQ(gate(two_payloads), nullptr) << "two payloads on one hand";
}

TEST(IdentityGate, AnyMismatchGetsNone) {
  auto epoch = bound_state();
  epoch.attached_objects.front().epoch = 3;
  EXPECT_EQ(gate(epoch), nullptr);
  auto frame = bound_state();
  frame.tracked_objects.frame_id = "map";
  EXPECT_EQ(gate(frame), nullptr);
  auto invalid = bound_state();
  invalid.tracked_objects_valid = false;
  EXPECT_EQ(gate(invalid), nullptr);
  auto free = bound_state();
  free.tracked_objects.objects.front().status = "confirmed";
  EXPECT_EQ(gate(free), nullptr);
  auto floor = bound_state();
  floor.tracked_objects.objects.front().floor_z = std::nan("");
  EXPECT_EQ(gate(floor), nullptr);
}

TEST(IdentityGate, ARefusalNamesTheConditionThatFailed) {
  // Spark R245/R253: the refusal logged no reason, and the stop it led to had to be guessed.
  const auto why = [](const WorldStateStamped& s) {
    const char* reason = "";
    EXPECT_EQ(bridge::identity_clearing_track(s.attached_objects.front(), s, "base_link", &reason),
              nullptr);
    return std::string(reason);
  };
  auto epoch = bound_state();
  epoch.attached_objects.front().epoch = 3;
  EXPECT_EQ(why(epoch), "epoch_mismatch");
  auto free = bound_state();
  free.tracked_objects.objects.front().status = "confirmed";
  EXPECT_EQ(why(free), "status_not_bound");
  auto floor = bound_state();
  floor.tracked_objects.objects.front().floor_z = std::nan("");
  EXPECT_EQ(why(floor), "no_floor_z");
  auto frame = bound_state();
  frame.tracked_objects.frame_id = "map";
  EXPECT_EQ(why(frame), "tracked_set_unusable");
  auto two_tracks = bound_state();
  two_tracks.tracked_objects.objects.push_back(track("trk:2", "bound", 0.16, 0.08, 0.03));
  EXPECT_EQ(why(two_tracks), "second_bound_on_hand");
  EXPECT_EQ(why(bound_state(bridge::kPinchFallbackEvidence)), "evidence_kind");
}

TEST(IdentityGeometry, TheTightHullClearsLessThanItsBox) {
  // A tetrahedron in the box's -x-y-z corner, with its 26-DOP.
  const double h = 0.05;
  const std::vector<tf2::Vector3> v{{-h, -h, -h}, {0.0, -h, -h}, {-h, 0.0, -h}, {-h, -h, 0.0}};
  auto object = attachment(bridge::kTrackedObjectEvidence, h);
  auto& prim = object.primitives.front();
  for (const auto& axis : bridge::kSelfDopAxis) {
    double lo = 1e9;
    double hi = -1e9;
    for (const auto& p : v) {
      const double d = axis[0] * p.x() + axis[1] * p.y() + axis[2] * p.z();
      lo = std::min(lo, d);
      hi = std::max(hi, d);
    }
    prim.tight_dop_lo.push_back(lo);
    prim.tight_dop_hi.push_back(hi);
  }
  for (const auto& p : v) {
    prim.tight_hull_vertices.insert(prim.tight_hull_vertices.end(), {p.x(), p.y(), p.z()});
  }
  std::vector<bridge::IdentityPiece> hull;
  ASSERT_TRUE(bridge::place_identity_pieces(object, tf2::Transform::getIdentity(), hull));
  auto plain = object;
  plain.primitives.front().tight_dop_lo.clear();
  plain.primitives.front().tight_dop_hi.clear();
  plain.primitives.front().tight_hull_vertices.clear();
  std::vector<bridge::IdentityPiece> box;
  ASSERT_TRUE(bridge::place_identity_pieces(plain, tf2::Transform::getIdentity(), box));

  const tf2::Vector3 far_corner(0.04, 0.04, 0.04);  // in the box, far from the hull
  EXPECT_TRUE(bridge::identity_covers(box, far_corner, 0.5 * kRes));
  EXPECT_FALSE(bridge::identity_covers(hull, far_corner, 0.5 * kRes));
  EXPECT_TRUE(bridge::identity_covers(hull, tf2::Vector3(-0.04, -0.04, -0.04), 0.5 * kRes));

  // Malformed refinement: clear nothing, never fall back to the box.
  prim.tight_dop_hi.pop_back();
  EXPECT_FALSE(bridge::place_identity_pieces(object, tf2::Transform::getIdentity(), hull));
  EXPECT_TRUE(hull.empty());
}

TEST(IdentityGrid, NeverBelowTheFloorNorIntoAnotherTracksBox) {
  openral_msgs::msg::OccupancyVoxels grid;
  grid.resolution = kRes;
  grid.size_x = grid.size_y = grid.size_z = 20;
  grid.orientation.w = 1.0;
  grid.occupancy.assign(20 * 20 * 20, 1);
  auto s = bound_state();
  s.tracked_objects.objects.front().floor_z = 0.06;
  // A packed neighbour whose box reaches into the payload.
  s.tracked_objects.objects.push_back(track("trk:2", "confirmed", 0.19, 0.08, 0.02));
  s.tracked_objects.objects.back().box.pose.position.y = 0.15;
  std::vector<bridge::IdentityPiece> pieces;
  tf2::Transform at(tf2::Quaternion::getIdentity(), tf2::Vector3(0.15, 0.15, 0.08));
  ASSERT_TRUE(bridge::place_identity_pieces(s.attached_objects.front(), at, pieces));
  bridge::IdentityKeepOut keep;
  ASSERT_TRUE(bridge::identity_keep_out(s.tracked_objects, *gate(s), tf2::Transform::getIdentity(),
                                        false, keep));
  ASSERT_GT(bridge::clear_identity_cells(grid, pieces, keep), 0U);
  for (std::size_t iz = 0; iz < 20; ++iz) {
    for (std::size_t iy = 0; iy < 20; ++iy) {
      for (std::size_t ix = 0; ix < 20; ++ix) {
        if (grid.occupancy[ix + 20 * (iy + 20 * iz)] != 0) {
          continue;
        }
        const tf2::Vector3 c((ix + 0.5) * kRes, (iy + 0.5) * kRes, (iz + 0.5) * kRes);
        EXPECT_GE(c.z(), 0.06) << "cleared below floor_z";
        EXPECT_LE((c - tf2::Vector3(0.15, 0.15, 0.08)).length(), 0.03 * std::sqrt(3.0) + kRes);
        const tf2::Vector3 d = c - tf2::Vector3(0.19, 0.15, 0.08);
        const double outside = tf2::Vector3(std::max(std::fabs(d.x()) - 0.02, 0.0),
                                            std::max(std::fabs(d.y()) - 0.02, 0.0),
                                            std::max(std::fabs(d.z()) - 0.02, 0.0))
                                   .length();
        EXPECT_GT(outside, 0.5 * kRes * std::sqrt(3.0)) << "cleared into the neighbour's box";
      }
    }
  }
}

TEST(IdentitySweep, PosesAreInterpolatedHalfACellApart) {
  std::vector<tf2::Transform> out;
  const tf2::Transform a(tf2::Quaternion::getIdentity(), tf2::Vector3(0.0, 0.0, 0.0));
  const tf2::Transform b(tf2::Quaternion::getIdentity(), tf2::Vector3(0.1, 0.0, 0.0));
  bridge::append_swept_poses(out, a, b, 0.05, 0.5 * kRes);
  ASSERT_EQ(out.size(), 14U);  // ceil(0.1 / 0.0075)
  EXPECT_NEAR(out.back().getOrigin().x(), 0.1, 1e-12);
  EXPECT_LE(out.front().getOrigin().x(), 0.5 * kRes + 1e-12);
}

TEST(IdentityOctree, OnlyAcceptedFinestLeavesAreClearedAndMerged) {
  octomap::OcTree tree(kRes);
  const auto occupy = [&](double x, double y, double z) {
    tree.updateNode(
        octomap::point3d(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)),
        true);
  };
  // A 3 x 2 slab of cells, plus one cell the predicate refuses.
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      occupy(0.0075 + i * kRes, 0.0075 + j * kRes, 0.0075);
    }
  }
  occupy(0.2, 0.0075, 0.0075);
  // A pruned 2x2x2 block: one coarser leaf, never cleared (clear_bbox would free it whole).
  for (int i = 0; i < 8; ++i) {
    occupy(0.12 + 0.0075 + (i & 1) * kRes, 0.0075 + ((i >> 1) & 1) * kRes,
           0.0075 + ((i >> 2) & 1) * kRes);
  }
  tree.prune();
  bridge::SweptBox window;
  window.add(tf2::Vector3(-0.05, -0.05, -0.05));
  window.add(tf2::Vector3(0.25, 0.1, 0.1));
  bool truncated = true;
  const auto boxes = bridge::octree_clear_boxes(
      tree, window, [](const tf2::Vector3& c) { return c.x() < 0.15; }, 64, truncated);
  EXPECT_FALSE(truncated);
  ASSERT_EQ(boxes.size(), 1U) << "the slab merges into one box";
  EXPECT_NEAR(boxes[0].lo.x(), 0.25 * kRes, 1e-6);
  EXPECT_NEAR(boxes[0].hi.x(), 2.75 * kRes, 1e-6);
  EXPECT_NEAR(boxes[0].hi.y(), 1.75 * kRes, 1e-6);
  EXPECT_NEAR(boxes[0].hi.z(), 0.75 * kRes, 1e-6);

  const auto all = bridge::octree_clear_boxes(
      tree, window, [](const tf2::Vector3&) { return true; }, 1, truncated);
  EXPECT_EQ(all.size(), 1U);
  EXPECT_TRUE(truncated) << "the cap leaves the rest in the octree";
}

// ── Release window (issue #418 F2) ──────────────────────────────────────────

constexpr std::int64_t kS = 1000000000;

// trk:1 RELEASED at x = 0.30 (away from the hand), the attachment still on the wire.
WorldStateStamped released_state(const std::string& evidence = bridge::kTrackedObjectEvidence) {
  auto s = bound_state(evidence);
  auto& t = s.tracked_objects.objects.front();
  t = track("trk:1", "released", 0.30, 0.08, 0.03);
  return s;
}

const bridge::IdentityKeys kWasBound{{"trk:1", 2}};

TEST(ReleaseClear, OpensOnlyForABindingTheBridgeWasClearing) {
  bridge::ReleaseClears w;
  const auto s = released_state();
  EXPECT_TRUE(bridge::advance_release_clears(w, {}, s, "base_link", 0, 3.0).empty())
      << "never bound by identity: nothing to clear";
  EXPECT_TRUE(w.empty());
  const auto open = bridge::advance_release_clears(w, kWasBound, s, "base_link", 0, 3.0);
  ASSERT_EQ(open.size(), 1U);
  EXPECT_EQ(open.front()->object_id, "trk:1");
  // Next tick: no longer bound, the window carries it.
  EXPECT_EQ(bridge::advance_release_clears(w, {}, s, "base_link", kS, 3.0).size(), 1U);
}

TEST(ReleaseClear, NothingWhileTheBindingLasts) {
  bridge::ReleaseClears w;
  EXPECT_TRUE(
      bridge::advance_release_clears(w, kWasBound, bound_state(), "base_link", 0, 3.0).empty());
  EXPECT_TRUE(w.empty());
}

TEST(ReleaseClear, ClosesOnReconfirmationAndNeverReopens) {
  bridge::ReleaseClears w;
  auto s = released_state();
  ASSERT_EQ(bridge::advance_release_clears(w, kWasBound, s, "base_link", 0, 3.0).size(), 1U);
  auto confirmed = s;
  confirmed.tracked_objects.objects.front().status = "confirmed";
  confirmed.tracked_objects.objects.front().epoch = 3;
  EXPECT_TRUE(bridge::advance_release_clears(w, {}, confirmed, "base_link", kS, 3.0).empty());
  EXPECT_TRUE(w.empty());
  EXPECT_TRUE(bridge::advance_release_clears(w, {}, s, "base_link", kS, 3.0).empty())
      << "released again without a binding the bridge cleared";
  // A new epoch while still RELEASED is another object state: closed too.
  ASSERT_EQ(bridge::advance_release_clears(w, kWasBound, s, "base_link", 0, 3.0).size(), 1U);
  auto lost = s;
  lost.tracked_objects.objects.front().status = "lost";
  EXPECT_TRUE(bridge::advance_release_clears(w, {}, lost, "base_link", kS, 3.0).empty());
  EXPECT_TRUE(w.empty());
}

TEST(ReleaseClear, ClosesOnTimeoutAndNeverReopens) {
  bridge::ReleaseClears w;
  const auto s = released_state();
  ASSERT_EQ(bridge::advance_release_clears(w, kWasBound, s, "base_link", 0, 3.0).size(), 1U);
  EXPECT_EQ(bridge::advance_release_clears(w, {}, s, "base_link", 3 * kS, 3.0).size(), 1U);
  EXPECT_TRUE(bridge::advance_release_clears(w, {}, s, "base_link", 3 * kS + 1, 3.0).empty());
  EXPECT_TRUE(bridge::advance_release_clears(w, {}, s, "base_link", 4 * kS, 3.0).empty());
}

TEST(ReleaseClear, OnlyWhileTheWireCarriesTheTrackedBinding) {
  // The kernel's frozen payload exists only while the wire carries (id, epoch): without
  // it, clearing the box would leave the object in neither the map nor the payload.
  bridge::ReleaseClears w;
  EXPECT_TRUE(bridge::advance_release_clears(
                  w, kWasBound, released_state(bridge::kPinchFallbackEvidence), "base_link", 0, 3.0)
                  .empty());
  auto dropped = released_state();
  dropped.attached_objects.clear();
  EXPECT_TRUE(bridge::advance_release_clears(w, kWasBound, dropped, "base_link", 0, 3.0).empty());
  auto epoch = released_state();
  epoch.attached_objects.front().epoch = 1;
  EXPECT_TRUE(bridge::advance_release_clears(w, kWasBound, epoch, "base_link", 0, 3.0).empty());
  auto twice = released_state();
  twice.attached_objects.push_back(twice.attached_objects.front());
  EXPECT_TRUE(bridge::advance_release_clears(w, kWasBound, twice, "base_link", 0, 3.0).empty());
  // Open, then the wire drops it: closed.
  ASSERT_EQ(
      bridge::advance_release_clears(w, kWasBound, released_state(), "base_link", 0, 3.0).size(),
      1U);
  EXPECT_TRUE(bridge::advance_release_clears(w, {}, dropped, "base_link", kS, 3.0).empty());
  EXPECT_TRUE(w.empty());
}

TEST(ReleaseClear, AnUnusableOrAmbiguousSetClearsNothing) {
  bridge::ReleaseClears w;
  auto dup = released_state();
  dup.tracked_objects.objects.push_back(dup.tracked_objects.objects.front());
  EXPECT_TRUE(bridge::advance_release_clears(w, kWasBound, dup, "base_link", 0, 3.0).empty());
  auto frame = released_state();
  frame.tracked_objects.frame_id = "map";
  EXPECT_TRUE(bridge::advance_release_clears(w, kWasBound, frame, "base_link", 0, 3.0).empty());
  auto floor = released_state();
  floor.tracked_objects.objects.front().floor_z = std::nan("");
  EXPECT_TRUE(bridge::advance_release_clears(w, kWasBound, floor, "base_link", 0, 3.0).empty());
  EXPECT_TRUE(w.empty());
}

TEST(ReleaseClear, TheReleasedBoxIsClearedWithEveryExclusion) {
  openral_msgs::msg::OccupancyVoxels grid;
  grid.resolution = kRes;
  grid.size_x = grid.size_y = grid.size_z = 30;
  grid.orientation.w = 1.0;
  grid.occupancy.assign(30 * 30 * 30, 1);
  auto s = released_state();
  auto& r = s.tracked_objects.objects.front();
  r.box.pose.position.x = r.box.pose.position.y = 0.2;
  r.floor_z = 0.06;
  s.tracked_objects.objects.push_back(track("trk:2", "confirmed", 0.24, 0.08, 0.02));
  s.tracked_objects.objects.back().box.pose.position.y = 0.2;
  std::vector<bridge::IdentityPiece> pieces;
  bridge::IdentityKeepOut keep;
  ASSERT_TRUE(bridge::released_clear_pieces(s.tracked_objects, s.tracked_objects.objects.front(),
                                            tf2::Transform::getIdentity(), pieces, keep));
  ASSERT_GT(bridge::clear_identity_cells(grid, pieces, keep), 0U);
  std::size_t inside_kept = 0;
  for (std::size_t iz = 0; iz < 30; ++iz) {
    for (std::size_t iy = 0; iy < 30; ++iy) {
      for (std::size_t ix = 0; ix < 30; ++ix) {
        const tf2::Vector3 c((ix + 0.5) * kRes, (iy + 0.5) * kRes, (iz + 0.5) * kRes);
        const tf2::Vector3 d = c - tf2::Vector3(0.2, 0.2, 0.08);
        const bool in_box = std::fabs(d.x()) <= 0.03 + 0.5 * kRes &&
                            std::fabs(d.y()) <= 0.03 + 0.5 * kRes &&
                            std::fabs(d.z()) <= 0.03 + 0.5 * kRes;
        if (grid.occupancy[ix + 30 * (iy + 30 * iz)] != 0) {
          const bool strictly_in =
              std::fabs(d.x()) <= 0.03 && std::fabs(d.y()) <= 0.03 && std::fabs(d.z()) <= 0.03;
          inside_kept += strictly_in && c.z() >= 0.06 && d.x() < 0.0 ? 1 : 0;
          continue;
        }
        EXPECT_TRUE(in_box) << "cleared outside the released box + half a cell";
        EXPECT_GE(c.z(), 0.06) << "cleared below floor_z";
        const tf2::Vector3 n = c - tf2::Vector3(0.24, 0.2, 0.08);
        const double outside = tf2::Vector3(std::max(std::fabs(n.x()) - 0.02, 0.0),
                                            std::max(std::fabs(n.y()) - 0.02, 0.0),
                                            std::max(std::fabs(n.z()) - 0.02, 0.0))
                                   .length();
        EXPECT_GT(outside, 0.5 * kRes * std::sqrt(3.0)) << "cleared into the neighbour's box";
      }
    }
  }
  EXPECT_EQ(inside_kept, 0U) << "the released box's far side was not cleared";
}

}  // namespace

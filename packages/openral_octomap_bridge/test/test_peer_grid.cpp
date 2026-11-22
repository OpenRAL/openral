// SPDX-License-Identifier: Apache-2.0
// The peer grid (ADR-0137): a peer's geometry, posed in this robot's base frame
// and swept along its motion, marks every cell it touches and nothing far away.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include <openral_msgs/msg/attached_collision_primitive.hpp>

#include "openral_octomap_bridge/peer_grid.hpp"

namespace ob = openral_octomap_bridge;

namespace {

// One link turning about z, a 50 mm sphere (a capsule of zero length) 0.5 m out on x.
ob::SelfModel arm() {
  ob::SelfModelParams p;
  p.n_links = 1;
  p.parent = {-1};
  p.joint_kind = {1};
  p.dof_index = {0};
  p.origin_xyzrpy = {0, 0, 0, 0, 0, 0};
  p.axis = {0, 0, 1};
  p.link_names = {"link"};
  p.capsule_link = {0};
  p.capsule_radius = {0.05};
  p.capsule_half_length = {0.0};
  p.capsule_origin_xyzrpy = {0.5, 0, 0, 0, 0, 0};
  ob::SelfModel model;
  std::string error;
  EXPECT_TRUE(ob::build_self_model(p, model, error)) << error;
  return model;
}

bool occupied(const ob::PeerGrid& g, const tf2::Vector3& p) {
  const auto i = [&](int a) {
    return static_cast<int>(std::floor((p[a] - g.origin[a]) / g.resolution));
  };
  const int ix = i(0);
  const int iy = i(1);
  const int iz = i(2);
  if (ix < 0 || iy < 0 || iz < 0 || ix >= g.n || iy >= g.n || iz >= g.n) {
    return false;
  }
  return g.occupancy[static_cast<std::size_t>(ix + g.n * (iy + g.n * iz))] != 0;
}

tf2::Transform at(double x, double y, double yaw) {
  tf2::Quaternion q;
  q.setRPY(0, 0, yaw);
  return tf2::Transform(q, tf2::Vector3(x, y, 0));
}

}  // namespace

TEST(PeerGrid, TheGridFitsTheKernelsCellCap) {
  // The launch's `_world_voxel_max_cells(1.05, 0.025)` is 85^3.
  EXPECT_EQ(ob::peer_cells_per_axis(1.05, 0.025), 85);
  const auto g = ob::empty_peer_grid(tf2::Vector3(0, 0, 0.5), 1.05, 0.025);
  EXPECT_EQ(g.occupancy.size(), 85U * 85U * 85U);
}

TEST(PeerGrid, APrimitiveMarksItsCellsAndNothingFar) {
  ob::PayloadPrimitive s;
  s.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_SPHERE;
  s.pose.setIdentity();
  s.pose.setOrigin(tf2::Vector3(0.3, 0.0, 0.5));
  s.radius = 0.05;
  auto g = ob::empty_peer_grid(tf2::Vector3(0, 0, 0.5), 1.0, 0.05);
  ASSERT_TRUE(ob::rasterize_primitives({s}, 0.0, g));
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.3, 0.0, 0.5)));
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.34, 0.0, 0.5))) << "the sphere's own surface";
  EXPECT_FALSE(occupied(g, tf2::Vector3(0.5, 0.0, 0.5)));
  EXPECT_FALSE(occupied(g, tf2::Vector3(-0.3, 0.0, 0.5)));
}

TEST(PeerGrid, APeerIsPosedThroughTheSharedWorld) {
  // This robot at world (0, 0); the peer at world (1, 0) facing back (-x), so its
  // sphere (0.5 m out on its own x) sits at world (0.5, 0): 0.5 m ahead of us.
  const auto model = arm();
  std::vector<ob::PayloadPrimitive> prims;
  ASSERT_TRUE(ob::peer_swept_primitives(model, {0.0}, {0.0}, {}, at(0, 0, 0), {}, at(1.0, 0, M_PI),
                                        {}, 0.3, prims));
  auto g = ob::empty_peer_grid(tf2::Vector3(0, 0, 0), 1.0, 0.05);
  ASSERT_TRUE(ob::rasterize_primitives(prims, 0.0, g));
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.5, 0.0, 0.0)));
  EXPECT_FALSE(occupied(g, tf2::Vector3(-0.5, 0.0, 0.0)));
}

TEST(PeerGrid, AMovingJointIsSweptOverTheHorizon) {
  // 2 rad/s for 0.3 s: the sphere ends 0.6 rad round, about (0.41, 0.28).
  const auto model = arm();
  std::vector<ob::PayloadPrimitive> prims;
  ASSERT_TRUE(ob::peer_swept_primitives(model, {0.0}, {2.0}, {}, tf2::Transform::getIdentity(), {},
                                        tf2::Transform::getIdentity(), {}, 0.3, prims));
  auto g = ob::empty_peer_grid(tf2::Vector3(0, 0, 0), 1.0, 0.025);
  ASSERT_TRUE(ob::rasterize_primitives(prims, 0.0, g));
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.5, 0.0, 0.0))) << "where it is";
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.5 * std::cos(0.3), 0.5 * std::sin(0.3), 0.0)))
      << "on the way";
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.5 * std::cos(0.6), 0.5 * std::sin(0.6), 0.0)))
      << "where it will be";
  EXPECT_FALSE(occupied(g, tf2::Vector3(0.5 * std::cos(-0.4), 0.5 * std::sin(-0.4), 0.0)))
      << "behind it";
}

TEST(PeerGrid, AMovingBaseIsSweptToo) {
  const auto model = arm();
  std::vector<ob::PayloadPrimitive> prims;
  // A fixed joint (base dof 0 held at zero) on a base driving +y at 1 m/s.
  ASSERT_TRUE(ob::peer_swept_primitives(model, {1.0}, {0.0}, {0}, tf2::Transform::getIdentity(), {},
                                        tf2::Transform::getIdentity(), {0.0, 1.0, 0.0}, 0.3,
                                        prims));
  auto g = ob::empty_peer_grid(tf2::Vector3(0, 0, 0), 1.0, 0.025);
  ASSERT_TRUE(ob::rasterize_primitives(prims, 0.0, g));
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.5, 0.0, 0.0))) << "a base dof is held at zero";
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.5, 0.15, 0.0)));
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.5, 0.3, 0.0)));
  EXPECT_FALSE(occupied(g, tf2::Vector3(0.5, 0.5, 0.0)));
}

TEST(PeerGrid, ThePlanarTwistRoundTrips) {
  const auto a = at(1.0, 2.0, 0.1);
  const auto b = at(1.05, 1.98, 0.14);
  const auto t = ob::planar_twist(a, b, 0.1);
  EXPECT_NEAR(t.vx, 0.5, 1e-9);
  EXPECT_NEAR(t.vy, -0.2, 1e-9);
  EXPECT_NEAR(t.wz, 0.4, 1e-9);
  const auto c = ob::advance_planar(a, t, 0.1);
  EXPECT_NEAR(c.getOrigin().distance(b.getOrigin()), 0.0, 1e-9);
  EXPECT_NEAR(c.getRotation().angleShortestPath(b.getRotation()), 0.0, 1e-9);
  // Across +-pi the yaw rate is the short way round.
  EXPECT_NEAR(ob::planar_twist(at(0, 0, 3.1), at(0, 0, -3.1), 1.0).wz, 2 * M_PI - 6.2, 1e-9);
}

TEST(PeerGrid, ThisRobotsOwnTravelIsSweptToo) {
  // The peer stands still at world (1, 0) facing back; this robot drives +x at 1 m/s.
  // Over 0.3 s the peer's sphere comes from 0.5 m to 0.2 m ahead of this robot's base.
  const auto model = arm();
  std::vector<ob::PayloadPrimitive> prims;
  ASSERT_TRUE(ob::peer_swept_primitives(model, {0.0}, {0.0}, {}, at(0, 0, 0), {1.0, 0.0, 0.0},
                                        at(1.0, 0, M_PI), {}, 0.3, prims));
  auto g = ob::empty_peer_grid(tf2::Vector3(0, 0, 0), 1.0, 0.025);
  ASSERT_TRUE(ob::rasterize_primitives(prims, 0.0, g));
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.5, 0.0, 0.0)));
  EXPECT_TRUE(occupied(g, tf2::Vector3(0.2, 0.0, 0.0))) << "where it will be, relative to us";
}

TEST(PeerGrid, AGarbageVelocityWithholdsTheGridInsteadOfUnderSweeping) {
  const auto model = arm();
  std::vector<ob::PayloadPrimitive> prims;
  EXPECT_FALSE(ob::peer_swept_primitives(model, {0.0}, {1e4}, {}, tf2::Transform::getIdentity(), {},
                                         tf2::Transform::getIdentity(), {}, 0.3, prims));
  EXPECT_TRUE(prims.empty());
  EXPECT_FALSE(ob::peer_swept_primitives(model, {0.0}, {0.0}, {}, tf2::Transform::getIdentity(), {},
                                         tf2::Transform::getIdentity(), {NAN, 0.0, 0.0}, 0.3,
                                         prims));
}

TEST(PeerGrid, ANonFinitePrimitiveIsRefusedNotDropped) {
  ob::PayloadPrimitive s;
  s.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_SPHERE;
  s.pose.setIdentity();
  s.pose.setOrigin(tf2::Vector3(NAN, 0.0, 0.0));
  s.radius = 0.05;
  auto g = ob::empty_peer_grid(tf2::Vector3(0, 0, 0), 1.0, 0.05);
  EXPECT_FALSE(ob::rasterize_primitives({s}, 0.0, g));
  // Huge but finite: outside the grid, no overflow, nothing marked.
  s.pose.setOrigin(tf2::Vector3(1e12, -1e12, 0.0));
  EXPECT_TRUE(ob::rasterize_primitives({s}, 0.0, g));
  EXPECT_EQ(std::count(g.occupancy.begin(), g.occupancy.end(), 1), 0);
}

TEST(PeerJoints, AStreamThatStopsIsStaleWhileAnotherCarriesOn) {
  ob::PeerJoints j(3);  // dofs 0, 1: the arm; 2: the gripper
  const std::vector<std::size_t> arm{0, 1, 2};
  j.update({2}, {0.04}, {}, 1'000'000'000);
  EXPECT_FALSE(j.complete(arm)) << "an arm never measured is not posed at zero";
  j.update({0, 1}, {0.5, -0.5}, {0.1, 0.0}, 1'000'000'000);
  ASSERT_TRUE(j.complete(arm));
  j.update({2}, {0.05}, {}, 5'000'000'000);  // only the gripper keeps publishing
  EXPECT_EQ(j.oldest_ns(arm), 1'000'000'000) << "the arm's own stamp, not the gripper's";
  EXPECT_EQ(j.newest_ns(arm), 5'000'000'000);
}

TEST(PeerJoints, AMessageWithoutVelocitiesKeepsTheOtherJointsVelocities) {
  ob::PeerJoints j(2);
  j.update({0}, {0.0}, {1.5}, 1'000'000'000);  // the arm, measured velocity
  j.update({1}, {0.0}, {}, 1'000'000'000);
  j.update({1}, {0.02}, {}, 1'100'000'000);  // the gripper, no velocity field
  EXPECT_DOUBLE_EQ(j.dq()[0], 1.5) << "the arm's velocity survives the gripper's message";
  EXPECT_NEAR(j.dq()[1], 0.2, 1e-9) << "the gripper's from its own previous sample";
  j.update({1}, {0.03}, {}, 1'100'000'500);  // 0.5 us later: too close to difference
  EXPECT_NEAR(j.dq()[1], 0.2, 1e-9);
}

TEST(PeerJoints, AStampFromTheFutureIsNotFresh) {
  EXPECT_TRUE(ob::within_age(0.1, 0.3));
  EXPECT_TRUE(ob::within_age(-0.1, 0.3)) << "a little clock jitter";
  EXPECT_FALSE(ob::within_age(0.4, 0.3)) << "stale";
  EXPECT_FALSE(ob::within_age(-0.4, 0.3)) << "from the future: would never age";
}

TEST(PeerJoints, JointsMeasuredAtDifferentTimesArePosedAtOneInstant) {
  ob::PeerJoints j(2);
  j.update({0}, {0.0}, {1.0}, 1'000'000'000);  // 1 rad/s, measured at 1.0 s
  j.update({1}, {0.5}, {0.0}, 1'200'000'000);  // still, measured at 1.2 s
  const auto q = j.q_at(1'300'000'000);
  EXPECT_NEAR(q[0], 0.3, 1e-9) << "0.3 s on from its own stamp";
  EXPECT_NEAR(q[1], 0.5, 1e-9);
}

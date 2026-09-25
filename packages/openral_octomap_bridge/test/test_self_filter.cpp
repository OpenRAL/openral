// SPDX-License-Identifier: Apache-2.0
// The robot self-filter's geometry: what it removes, what it keeps, and that its
// forward kinematics is the safety kernel's, operation for operation.
//
// The FK is a deliberate mirror of `openral_safety_kernel::forward_kinematics`
// (the filter is Layer 2, the kernel Layer 5; see self_filter.hpp). This test is
// the one place the two meet: it links the kernel library and requires every
// link frame to agree to 1e-12 on a chain that exercises every joint kind, a
// non-trivial RPY origin and a non-axis-aligned revolute axis.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <openral_safety_kernel/collision.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <openral_msgs/msg/attached_collision_primitive.hpp>

#include "openral_octomap_bridge/self_filter.hpp"

namespace bridge = openral_octomap_bridge;
namespace osk = openral_safety_kernel;
using Wire = openral_msgs::msg::AttachedCollisionPrimitive;

namespace {

// root (fixed) -> link1 (revolute about a tilted axis) -> link2 (prismatic
// along x) -> link3 (revolute about z), each with an RPY origin.
bridge::SelfModelParams chain_params() {
  bridge::SelfModelParams p;
  p.n_links = 4;
  p.parent = {-1, 0, 1, 2};
  p.joint_kind = {0, 1, 2, 1};
  p.dof_index = {-1, 0, 1, 2};
  p.origin_xyzrpy = {0,    0,    0,     0,    0,    0,    //
                     0.1,  -0.2, 0.3,   0.2,  -0.1, 0.4,  //
                     0.0,  0.25, 0.0,   0.0,  0.3,  0.0,  //
                     0.05, 0.0,  -0.15, -0.5, 0.0,  1.1};
  const double s = 1.0 / std::sqrt(3.0);
  p.axis = {0, 0, 1, s, s, s, 1, 0, 0, 0, 0, 1};
  p.link_names = {"root", "l1", "l2", "l3"};
  p.capsule_link = {1, 3};
  p.capsule_radius = {0.05, 0.04};
  p.capsule_half_length = {0.1, 0.0};  // a capsule and a sphere
  p.capsule_origin_xyzrpy = {0, 0, -0.1, 0, 0, 0, 0, 0, -0.04, 0, 0, 0};
  return p;
}

osk::CollisionModel kernel_model(const bridge::SelfModelParams& p) {
  osk::CollisionModel m;
  m.n_links = static_cast<std::size_t>(p.n_links);
  for (std::size_t i = 0; i < m.n_links; ++i) {
    m.parent.push_back(static_cast<int>(p.parent[i]));
    m.joint_kind.push_back(static_cast<osk::JointKind>(static_cast<std::uint8_t>(p.joint_kind[i])));
    m.dof_index.push_back(static_cast<int>(p.dof_index[i]));
    m.origin.push_back(osk::transform_from_xyz_rpy(
        p.origin_xyzrpy[6 * i + 0], p.origin_xyzrpy[6 * i + 1], p.origin_xyzrpy[6 * i + 2],
        p.origin_xyzrpy[6 * i + 3], p.origin_xyzrpy[6 * i + 4], p.origin_xyzrpy[6 * i + 5]));
    m.axis.push_back(osk::Vec3{p.axis[3 * i + 0], p.axis[3 * i + 1], p.axis[3 * i + 2]});
  }
  return m;
}

// Pack xyz points as FLOAT32 records with a 4-byte pad (point_step 16), like a
// PointCloud2 with x,y,z,rgb.
std::vector<std::uint8_t> pack(const std::vector<tf2::Vector3>& pts) {
  std::vector<std::uint8_t> out(pts.size() * 16, 0);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const float xyz[3] = {static_cast<float>(pts[i].x()), static_cast<float>(pts[i].y()),
                          static_cast<float>(pts[i].z())};
    std::memcpy(out.data() + i * 16, xyz, sizeof(xyz));
  }
  return out;
}

}  // namespace

TEST(SelfFilterFk, MatchesTheSafetyKernelForwardKinematicsExactly) {
  const auto params = chain_params();
  bridge::SelfModel model;
  std::string error;
  ASSERT_TRUE(bridge::build_self_model(params, model, error)) << error;
  const auto kernel = kernel_model(params);
  osk::CollisionScratch scratch;
  scratch.link_world.resize(kernel.n_links);
  std::vector<tf2::Transform> ours;

  for (const auto& q : std::vector<std::vector<double>>{
           {0.0, 0.0, 0.0}, {0.7, 0.12, -1.3}, {-2.1, -0.05, 2.9}, {3.1, 0.4, 0.01}}) {
    osk::forward_kinematics(kernel, q.data(), q.size(), scratch);
    bridge::self_forward_kinematics(model, q.data(), q.size(), ours);
    for (std::size_t i = 0; i < kernel.n_links; ++i) {
      const auto& k = scratch.link_world[i];
      const tf2::Matrix3x3& r = ours[i].getBasis();
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          EXPECT_NEAR(r[row][col], k.r[row * 3 + col], 1e-12) << "link " << i << " r" << row << col;
        }
      }
      EXPECT_NEAR(ours[i].getOrigin().x(), k.t.x, 1e-12) << "link " << i;
      EXPECT_NEAR(ours[i].getOrigin().y(), k.t.y, 1e-12) << "link " << i;
      EXPECT_NEAR(ours[i].getOrigin().z(), k.t.z, 1e-12) << "link " << i;
    }
  }
}

TEST(SelfFilterModel, RefusesShapesTheKernelWouldRefuse) {
  std::string error;
  bridge::SelfModel model;
  auto p = chain_params();
  p.origin_xyzrpy.pop_back();
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
  p = chain_params();
  p.parent[2] = 3;  // child before parent
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
  p = chain_params();
  p.capsule_link[0] = 9;
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
  p = chain_params();
  p.capsule_radius[0] = -0.01;
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
  p = chain_params();
  p.joint_kind[1] = 7;
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
  EXPECT_EQ(model.n_links(), 0U) << "a refused model must leave nothing behind";
}

TEST(SelfFilterMask, RemovesTheSurfaceAndThePaddingShellKeepsWhatIsOutside) {
  // One capsule, r = 0.05, half length 0.1, centred at (1, 0, 0), axis along z.
  bridge::PayloadPrimitive cap;
  cap.shape_type = Wire::SHAPE_CAPSULE;
  cap.radius = 0.05;
  cap.half_length = 0.1;
  cap.pose.setOrigin(tf2::Vector3(1.0, 0.0, 0.0));
  bridge::SelfMask mask;
  mask.set({cap}, 0.03);

  const std::vector<tf2::Vector3> removed = {
      {1.0, 0.0, 0.0},    // axis
      {1.05, 0.0, 0.05},  // on the surface
      {1.079, 0.0, 0.0},  // inside the 3 cm shell
      {1.0, 0.0, 0.179},  // past the end cap, inside the shell
  };
  const std::vector<tf2::Vector3> kept = {
      {1.081, 0.0, 0.0},  // just outside the shell
      {1.0, 0.0, 0.181},
      {0.0, 0.0, 0.0},
  };
  for (const auto& p : removed) {
    EXPECT_TRUE(mask.contains(p)) << p.x() << "," << p.y() << "," << p.z();
  }
  for (const auto& p : kept) {
    EXPECT_FALSE(mask.contains(p)) << p.x() << "," << p.y() << "," << p.z();
  }

  std::vector<tf2::Vector3> all = removed;
  all.insert(all.end(), kept.begin(), kept.end());
  auto data = pack(all);
  // One NaN point, the ZED's "no return".
  const float nan = std::nanf("");
  std::memcpy(data.data() + 2 * 16, &nan, sizeof(float));
  std::vector<std::uint8_t> out;
  const auto stats = bridge::filter_xyz_points(data.data(), all.size(), 16, 0, 4, 8, mask, out);
  EXPECT_EQ(stats.points_in, all.size());
  EXPECT_EQ(stats.non_finite, 1U);
  EXPECT_EQ(stats.removed, removed.size() - 1);
  EXPECT_EQ(stats.kept, kept.size());
  ASSERT_EQ(out.size(), kept.size() * 16);
  float first[3];
  std::memcpy(first, out.data(), sizeof(first));
  EXPECT_FLOAT_EQ(first[0], 1.081F) << "kept records are copied whole, in order";
}

TEST(SelfFilterPlacement, PrimitivesFollowTheArmAndTheCameraFrame) {
  // The sphere on link 3 must be removed where FK puts it at this q, seen from
  // a camera frame rotated and offset from the root, and nowhere else.
  const auto params = chain_params();
  bridge::SelfModel model;
  std::string error;
  ASSERT_TRUE(bridge::build_self_model(params, model, error)) << error;
  const std::vector<double> q = {0.9, 0.1, -0.6};
  std::vector<tf2::Transform> link_world;
  bridge::self_forward_kinematics(model, q.data(), q.size(), link_world);

  const tf2::Transform camera_from_root =
      bridge::self_transform_from_xyz_rpy(0.3, -0.1, 0.8, 0.4, 1.2, -0.3);
  std::vector<bridge::PayloadPrimitive> placed;
  bridge::place_self_primitives(model, link_world, camera_from_root, placed);
  ASSERT_EQ(placed.size(), 2U);
  bridge::SelfMask mask;
  mask.set(placed, 0.0);

  const tf2::Vector3 sphere_center_root = link_world[3] * tf2::Vector3(0.0, 0.0, -0.04);
  const tf2::Vector3 in_camera = camera_from_root * sphere_center_root;
  EXPECT_TRUE(mask.contains(in_camera)) << "the sphere's centre, seen from the camera";
  EXPECT_FALSE(mask.contains(in_camera + tf2::Vector3(0.0, 0.0, 0.2)))
      << "20 cm off the sphere is world, not robot";
  EXPECT_FALSE(mask.contains(sphere_center_root) && !mask.contains(in_camera))
      << "placement must use the camera frame";
}

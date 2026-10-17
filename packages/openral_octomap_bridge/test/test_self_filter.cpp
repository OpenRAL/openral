// SPDX-License-Identifier: Apache-2.0
// The robot self-filter's geometry: what it removes, what it keeps, and that its
// forward kinematics is the safety kernel's, operation for operation.
//
// The FK is a deliberate mirror of `openral_safety_kernel::forward_kinematics`
// (the filter is Layer 2, the kernel Layer 5; see self_filter.hpp). This test is
// the one place the two meet: it links the kernel library and requires every
// link frame to agree to 1e-12 on a chain that exercises every joint kind, a
// non-trivial RPY origin and a non-axis-aligned revolute axis. The point-vs-hull
// distance is pinned the same way, against the kernel's staged narrow phase
// with the voxel cube shrunk to a point, on the real panda_mobile hulls.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <openral_safety_kernel/collision.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <yaml-cpp/yaml.h>

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
  m.mimic_multiplier = p.mimic_multiplier;
  m.mimic_offset = p.mimic_offset;
  return m;
}

// `chain_params()` plus a fifth link: a mimic-revolute follower of column 1
// (l2's prismatic column) hung off l2, `q = -2 * q[1] + 0.3` about its own axis
// — the kernel's `JointKind::kMimicRevolute`, as the OpenArm's follower finger.
bridge::SelfModelParams chain_with_mimic_params() {
  auto p = chain_params();
  p.n_links = 5;
  p.parent.push_back(2);
  p.joint_kind.push_back(3);
  p.dof_index.push_back(1);
  p.origin_xyzrpy.insert(p.origin_xyzrpy.end(), {0.02, -0.01, 0.0, 0.0, 0.0, 0.2});
  p.axis.insert(p.axis.end(), {0.0, 1.0, 0.0});
  p.link_names.push_back("follower");
  p.mimic_multiplier = {1.0, 1.0, 1.0, 1.0, -2.0};
  p.mimic_offset = {0.0, 0.0, 0.0, 0.0, 0.3};
  return p;
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

TEST(SelfFilterFk, AMimicLinkMatchesTheKernelForwardKinematicsExactly) {
  const auto params = chain_with_mimic_params();
  bridge::SelfModel model;
  std::string error;
  ASSERT_TRUE(bridge::build_self_model(params, model, error)) << error;
  const auto kernel = kernel_model(params);
  osk::CollisionScratch scratch;
  scratch.link_world.resize(kernel.n_links);
  std::vector<tf2::Transform> ours;
  for (const auto& q :
       std::vector<std::vector<double>>{{0.0, 0.0, 0.0}, {0.7, 0.12, -1.3}, {-2.1, -0.05, 2.9}}) {
    osk::forward_kinematics(kernel, q.data(), q.size(), scratch);
    bridge::self_forward_kinematics(model, q.data(), q.size(), ours);
    const auto& k = scratch.link_world[4];
    const tf2::Matrix3x3& r = ours[4].getBasis();
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        EXPECT_NEAR(r[row][col], k.r[row * 3 + col], 1e-12) << "r" << row << col;
      }
    }
    EXPECT_NEAR(ours[4].getOrigin().x(), k.t.x, 1e-12);
    EXPECT_NEAR(ours[4].getOrigin().y(), k.t.y, 1e-12);
    EXPECT_NEAR(ours[4].getOrigin().z(), k.t.z, 1e-12);
    // And it is the mimic relation, not the raw column: rotation about y by
    // -2 * q[1] + 0.3 shows in the follower's basis relative to its parent.
    const tf2::Transform rel = ours[2].inverse() * ours[4];
    const double theta = -2.0 * q[1] + 0.3;
    const tf2::Matrix3x3 expected = tf2::Matrix3x3(tf2::Quaternion(tf2::Vector3(0, 0, 1), 0.2)) *
                                    tf2::Matrix3x3(tf2::Quaternion(tf2::Vector3(0, 1, 0), theta));
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        EXPECT_NEAR(rel.getBasis()[row][col], expected[row][col], 1e-12) << "rel r" << row << col;
      }
    }
  }
}

TEST(SelfFilterModel, RefusesAMimicLinkTheKernelWouldRefuse) {
  std::string error;
  bridge::SelfModel model;
  auto p = chain_with_mimic_params();
  p.mimic_offset.pop_back();  // not per link
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
  p = chain_with_mimic_params();
  p.mimic_multiplier[4] = 0.0;  // a fixed link wearing a moving kind
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
  p = chain_with_mimic_params();
  p.dof_index[4] = -1;  // no column to read
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
  p = chain_with_mimic_params();
  p.joint_kind[4] = 5;  // past kMimicPrismatic
  EXPECT_FALSE(bridge::build_self_model(p, model, error));
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

// ---------------------------------------------------------------------------
// Tight geometry: a box refined by the kernel's hull.
// ---------------------------------------------------------------------------

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// The kernel's own reading of a hull: its `LinkHull` + vertex buffer.
struct KernelHull {
  osk::LinkHull hull;
  std::vector<osk::Vec3> vertices;
};

KernelHull kernel_hull(const bridge::SelfHull& h) {
  KernelHull k;
  for (int i = 0; i < osk::kDopAxes; ++i) {
    k.hull.dop_lo[i] = h.dop_lo[i];
    k.hull.dop_hi[i] = h.dop_hi[i];
  }
  for (const auto& v : h.vertices) {
    k.vertices.push_back(osk::Vec3{v.x(), v.y(), v.z()});
  }
  k.hull.vertex_first = 0;
  k.hull.vertex_count = static_cast<int>(k.vertices.size());
  return k;
}

// The kernel's staged narrow phase for a zero-size cell at `p`, box at the
// identity: stage 1, then stage 2 with stage 1 as its fallback.
double kernel_point_distance(const KernelHull& k, const tf2::Vector3& p) {
  osk::TightPose pose;
  osk::tight_pose_init(k.hull, k.vertices.empty() ? nullptr : k.vertices.data(), osk::Transform{},
                       0.0, pose);
  const osk::Vec3 c{p.x(), p.y(), p.z()};
  osk::Vec3 seed{};
  const double stage1 = osk::dop_cell_lower_bound(pose, c, 0.0, &seed);
  osk::GjkWitness witness;
  return osk::hull_cell_distance(pose, c, 0.0, seed, kInfinity, stage1, witness);
}

// The 26-DOP of a vertex set: tangent slabs, as tools/generate_tight_geometry.py
// builds them.
void fit_dop(bridge::SelfHull& h) {
  for (int i = 0; i < bridge::kSelfDopAxes; ++i) {
    const tf2::Vector3 u(bridge::kSelfDopAxis[i][0], bridge::kSelfDopAxis[i][1],
                         bridge::kSelfDopAxis[i][2]);
    h.dop_lo[i] = kInfinity;
    h.dop_hi[i] = -kInfinity;
    for (const auto& v : h.vertices) {
      h.dop_lo[i] = std::min(h.dop_lo[i], u.dot(v));
      h.dop_hi[i] = std::max(h.dop_hi[i], u.dot(v));
    }
  }
}

// One link carrying one 0.2 m box refined by `hull` (vertex list; DOP fitted).
bridge::SelfModelParams one_box_params(const std::vector<tf2::Vector3>& hull_vertices,
                                       const tf2::Vector3& half_extents) {
  bridge::SelfModelParams p;
  p.n_links = 1;
  p.parent = {-1};
  p.joint_kind = {0};
  p.dof_index = {-1};
  p.origin_xyzrpy = {0, 0, 0, 0, 0, 0};
  p.axis = {0, 0, 1};
  p.link_names = {"root"};
  p.box_link = {0};
  p.box_half_extents = {half_extents.x(), half_extents.y(), half_extents.z()};
  p.box_origin_xyzrpy = {0, 0, 0, 0, 0, 0};
  bridge::SelfHull h;
  h.vertices = hull_vertices;
  fit_dop(h);
  p.box_hull = {0};
  p.hull_dop_lo.assign(h.dop_lo, h.dop_lo + bridge::kSelfDopAxes);
  p.hull_dop_hi.assign(h.dop_hi, h.dop_hi + bridge::kSelfDopAxes);
  p.hull_vertex_first = {0};
  p.hull_vertex_count = {static_cast<std::int64_t>(hull_vertices.size())};
  for (const auto& v : hull_vertices) {
    p.hull_vertices.insert(p.hull_vertices.end(), {v.x(), v.y(), v.z()});
  }
  return p;
}

// The regular tetrahedron inscribed in the 0.2 m cube. Its face normals are
// DOP corner axes, so stage 1 alone is exact on its faces.
std::vector<tf2::Vector3> tetrahedron() {
  return {{0.1, 0.1, 0.1}, {0.1, -0.1, -0.1}, {-0.1, 0.1, -0.1}, {-0.1, -0.1, 0.1}};
}

// An irregular tetrahedron: no face normal is a DOP axis, so its 26-DOP is
// strictly larger and only stage 2 measures its faces exactly.
std::vector<tf2::Vector3> irregular_tetrahedron() {
  return {{0.1, 0.02, 0.0}, {-0.03, 0.09, 0.01}, {-0.06, -0.07, 0.05}, {0.01, -0.02, -0.1}};
}

bridge::SelfMask mask_of(const bridge::SelfModel& model, double padding, bool with_hulls) {
  std::vector<tf2::Transform> link_world;
  const double q = 0.0;
  bridge::self_forward_kinematics(model, &q, 0, link_world);
  std::vector<bridge::PayloadPrimitive> placed;
  std::vector<const bridge::SelfHull*> hulls;
  bridge::place_self_primitives(model, link_world, tf2::Transform::getIdentity(), placed, &hulls);
  bridge::SelfMask mask;
  mask.set(placed, padding, with_hulls ? hulls : std::vector<const bridge::SelfHull*>{});
  return mask;
}

std::filesystem::path repo_root() { return std::filesystem::path(OPENRAL_REPO_ROOT); }

}  // namespace

TEST(SelfFilterHull, ConstantsAreTheKernels) {
  ASSERT_EQ(bridge::kSelfDopAxes, osk::kDopAxes);
  for (int i = 0; i < osk::kDopAxes; ++i) {
    for (int k = 0; k < 3; ++k) {
      EXPECT_EQ(bridge::kSelfDopAxis[i][k], osk::kDopAxis[i][k]) << "axis " << i;
    }
  }
  EXPECT_EQ(bridge::kSelfMaxHullVertices, osk::kMaxTightHullVertices);
  EXPECT_EQ(bridge::kSelfHullContainmentEpsilonM, osk::kTightContainmentEpsilonM);
  EXPECT_EQ(bridge::kSelfGjkMaxIterations, osk::kGjkMaxIterations);
  EXPECT_EQ(bridge::kSelfGjkTolerance, osk::kGjkTolerance);
}

TEST(SelfFilterHull, DistanceIsExactOnATetrahedronAndNeverAnOverReport) {
  bridge::SelfHull h;
  h.vertices = tetrahedron();
  fit_dop(h);
  // Face through (0.1,0.1,0.1), (0.1,-0.1,-0.1), (-0.1,0.1,-0.1): outward
  // normal (1,1,-1)/sqrt(3), offset 0.1/sqrt(3). A point 3 cm along it from
  // the face centroid is exactly 3 cm from the hull.
  const tf2::Vector3 n = tf2::Vector3(1, 1, -1).normalized();
  const tf2::Vector3 centroid = (h.vertices[0] + h.vertices[1] + h.vertices[2]) / 3.0;
  EXPECT_NEAR(bridge::self_hull_distance(h, centroid + n * 0.03, kInfinity), 0.03, 1e-9);
  EXPECT_LE(bridge::self_hull_distance(h, tf2::Vector3(0, 0, 0), kInfinity), 0.0) << "inside";
  // Off a vertex: the distance is the distance to that vertex.
  const tf2::Vector3 off = h.vertices[0] + tf2::Vector3(1, 1, 1).normalized() * 0.05;
  EXPECT_NEAR(bridge::self_hull_distance(h, off, kInfinity), 0.05, 1e-9);

  // Random points: never farther than the nearest vertex (the hull contains
  // every vertex), and equal to the kernel's staged narrow phase.
  const KernelHull k = kernel_hull(h);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-0.3, 0.3);
  for (int i = 0; i < 2000; ++i) {
    const tf2::Vector3 p(u(rng), u(rng), u(rng));
    const double d = bridge::self_hull_distance(h, p, kInfinity);
    double nearest_vertex = kInfinity;
    for (const auto& v : h.vertices) {
      nearest_vertex = std::min(nearest_vertex, (p - v).length());
    }
    EXPECT_LE(d, nearest_vertex + 1e-12);
    EXPECT_NEAR(d, kernel_point_distance(k, p), 1e-9) << p.x() << "," << p.y() << "," << p.z();
  }
}

TEST(SelfFilterHull, BoxCornerSlackIsKeptWhenTheBoxCarriesAHull) {
  const auto verts = irregular_tetrahedron();
  bridge::SelfModel model;
  std::string error;
  ASSERT_TRUE(bridge::build_self_model(one_box_params(verts, {0.1, 0.1, 0.1}), model, error))
      << error;
  ASSERT_EQ(model.primitive_hull, std::vector<int>{0});
  const double padding = 0.02;
  const auto hull_mask = mask_of(model, padding, true);
  const auto box_mask = mask_of(model, padding, false);

  // A box corner the hull does not reach: on the box, far from the hull.
  const tf2::Vector3 corner(0.1, 0.1, -0.1);
  ASSERT_GT(bridge::self_hull_distance(model.hulls[0], corner, kInfinity), 0.05);
  EXPECT_TRUE(box_mask.contains(corner)) << "the box-only filter removes it";
  EXPECT_FALSE(hull_mask.contains(corner)) << "the box's corner slack is not the robot";
  EXPECT_TRUE(hull_mask.contains((verts[0] + verts[1] + verts[2] + verts[3]) / 4.0))
      << "inside the hull";

  // Off the face (v0, v1, v2) along its outward normal from its centroid: the
  // centroid is the nearest hull point, so the offset IS the distance. No DOP
  // axis is normal to this face, so this is stage 2's answer.
  tf2::Vector3 n = (verts[1] - verts[0]).cross(verts[2] - verts[0]).normalized();
  if (n.dot(verts[3] - verts[0]) > 0.0) {
    n = -n;
  }
  const tf2::Vector3 centroid = (verts[0] + verts[1] + verts[2]) / 3.0;
  EXPECT_NEAR(bridge::self_hull_distance(model.hulls[0], centroid + n * 0.03, kInfinity), 0.03,
              1e-9);
  EXPECT_TRUE(hull_mask.contains(centroid + n * 0.019)) << "inside the padding of the hull";
  EXPECT_FALSE(hull_mask.contains(centroid + n * 0.021)) << "past the padding of the hull";
}

TEST(SelfFilterModel, RefusesAMalformedHull) {
  const auto good = one_box_params(tetrahedron(), {0.1, 0.1, 0.1});
  bridge::SelfModel model;
  std::string error;
  ASSERT_TRUE(bridge::build_self_model(good, model, error)) << error;

  const auto refuses = [&](bridge::SelfModelParams p, const char* what) {
    bridge::SelfModel m;
    std::string e;
    EXPECT_FALSE(bridge::build_self_model(p, m, e)) << what;
    EXPECT_FALSE(e.empty()) << what;
    EXPECT_EQ(m.n_links(), 0U) << what;
  };
  auto p = good;
  p.box_hull = {0, 0};
  refuses(p, "box_hull not parallel to the boxes");
  p = good;
  p.box_hull = {1};
  refuses(p, "box_hull names a missing hull");
  p = good;
  p.box_hull = {-2};
  refuses(p, "box_hull below -1");
  p = good;
  p.hull_dop_lo.pop_back();
  refuses(p, "DOP arity");
  p = good;
  p.hull_vertices.pop_back();
  refuses(p, "vertex buffer not xyz triples");
  p = good;
  p.hull_vertex_count = {5};
  refuses(p, "vertex slice past the buffer");
  p = good;
  p.hull_vertex_first = {-1};
  refuses(p, "negative vertex offset");
  p = good;
  p.hull_dop_hi[5] = std::nan("");
  refuses(p, "non-finite slab");
  p = good;
  std::swap(p.hull_dop_lo[4], p.hull_dop_hi[4]);
  refuses(p, "inverted slab");
  p = good;
  p.box_half_extents = {0.09, 0.1, 0.1};
  refuses(p, "DOP escapes the box");
  p = good;
  p.hull_dop_hi[7] -= 0.01;
  refuses(p, "a vertex outside its own DOP");
  p = good;
  p.hull_vertices[0] = std::nan("");
  refuses(p, "non-finite vertex");
  p = good;
  std::fill(p.hull_dop_lo.begin(), p.hull_dop_lo.end(), 0.0);
  std::fill(p.hull_dop_hi.begin(), p.hull_dop_hi.end(), 0.0);
  p.hull_vertex_count = {0};
  refuses(p, "a DOP collapsed to a point");
  p = good;
  p.hull_vertex_count = {bridge::kSelfMaxHullVertices + 1};
  p.hull_vertices.resize(3 * static_cast<std::size_t>(bridge::kSelfMaxHullVertices + 1), 0.0);
  refuses(p, "over the kernel's vertex budget");
  p = good;
  p.box_hull.clear();
  refuses(p, "hull arrays without box_hull");
}

TEST(SelfFilterManifest, PandaMobileHullsLoadAndOnlyEverShrinkTheShell) {
  // The real manifest: every panda_mobile arm link is a box with the kernel's
  // tight geometry, link1 as a DOP only (its hull is over the vertex budget).
  const auto path = repo_root() / "robots" / "panda_mobile" / "robot.yaml";
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "no " << path << " (not a source checkout)";
  }
  const YAML::Node robot = YAML::LoadFile(path.string());
  const double padding = 0.02;
  int links = 0;
  int exact = 0;
  for (const auto& geom : robot["collision_geometry"]) {
    const auto tight = geom["tight_geometry"];
    if (!tight || geom["shape"]["shape"].as<std::string>() != "box") {
      continue;
    }
    const auto he = geom["shape"]["half_extents_m"].as<std::vector<double>>();
    bridge::SelfModelParams p = one_box_params({}, {he[0], he[1], he[2]});
    p.hull_dop_lo = tight["dop_lo_m"].as<std::vector<double>>();
    p.hull_dop_hi = tight["dop_hi_m"].as<std::vector<double>>();
    p.hull_vertices.clear();
    for (const auto& v : tight["hull_vertices_m"]) {
      const auto xyz = v.as<std::vector<double>>();
      p.hull_vertices.insert(p.hull_vertices.end(), xyz.begin(), xyz.end());
    }
    p.hull_vertex_count = {static_cast<std::int64_t>(p.hull_vertices.size() / 3)};
    const auto link = geom["link_name"].as<std::string>();
    bridge::SelfModel model;
    std::string error;
    ASSERT_TRUE(bridge::build_self_model(p, model, error)) << link << ": " << error;
    ++links;
    exact += model.hulls[0].vertices.empty() ? 0 : 1;

    const auto hull_mask = mask_of(model, padding, true);
    const auto box_mask = mask_of(model, padding, false);
    const KernelHull k = kernel_hull(model.hulls[0]);
    std::mt19937 rng(static_cast<unsigned>(links));
    std::uniform_real_distribution<double> ux(-he[0] - padding, he[0] + padding);
    std::uniform_real_distribution<double> uy(-he[1] - padding, he[1] + padding);
    std::uniform_real_distribution<double> uz(-he[2] - padding, he[2] + padding);
    int box_removed = 0;
    int hull_removed = 0;
    for (int i = 0; i < 4000; ++i) {
      const tf2::Vector3 pt(ux(rng), uy(rng), uz(rng));
      const bool by_box = box_mask.contains(pt);
      const bool by_hull = hull_mask.contains(pt);
      box_removed += by_box ? 1 : 0;
      hull_removed += by_hull ? 1 : 0;
      EXPECT_TRUE(by_box || !by_hull) << link << ": the hull removed a point the box keeps";
      EXPECT_NEAR(bridge::self_hull_distance(model.hulls[0], pt, kInfinity),
                  kernel_point_distance(k, pt), 1e-9)
          << link;
    }
    EXPECT_LT(hull_removed, box_removed) << link << ": the box's slack must be handed back";
  }
  EXPECT_EQ(links, 7) << "panda_link1..7";
  EXPECT_EQ(exact, 6) << "every link but panda_link1 carries its exact hull";
}

// ---------------------------------------------------------------------------
// Every real manifest with mimic joints: filter FK == kernel FK over the stroke.
// ---------------------------------------------------------------------------

namespace {

std::vector<double> seq_or(const YAML::Node& n, std::vector<double> fallback) {
  return n ? n.as<std::vector<double>>() : fallback;
}

// The kinematic lowering of `envelope_loader._ordered_collision_links`: joints,
// fixed attachments and mimic joints as one tree, breadth-first from the root
// with children in name order. Both FKs read these same arrays, so the pin is
// on the C++ mirrors, not on this lowering.
bridge::SelfModelParams manifest_chain(const YAML::Node& robot,
                                       std::vector<std::pair<double, double>>& limits,
                                       std::vector<int>& mimicked) {
  struct Edge {
    std::string parent;
    std::int64_t kind{0};
    std::int64_t dof{-1};
    std::vector<double> xyzrpy;
    std::vector<double> axis;
    double mult{1.0};
    double off{0.0};
  };
  std::map<std::string, Edge> edge_of;
  std::map<std::string, int> column;
  int idx = 0;
  for (const auto& j : robot["joints"]) {
    const auto type = j["joint_type"].as<std::string>();
    Edge e;
    e.parent = j["parent_link"].as<std::string>();
    e.kind = type == "prismatic" ? 2 : type == "fixed" ? 0 : 1;
    e.dof = idx;
    const auto xyz = seq_or(j["origin_xyz"], {0, 0, 0});
    const auto rpy = seq_or(j["origin_rpy"], {0, 0, 0});
    e.xyzrpy = {xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]};
    e.axis = seq_or(j["axis_xyz"], {0, 0, 1});
    const auto lim = seq_or(j["position_limits"], {-1.0, 1.0});
    limits.emplace_back(lim[0], lim[1]);
    column[j["name"].as<std::string>()] = idx++;
    edge_of[j["child_link"].as<std::string>()] = e;
  }
  const YAML::Node fixed = robot["fixed_attachments"] ? robot["fixed_attachments"]
                                                      : YAML::Node(YAML::NodeType::Sequence);
  for (const auto& a : fixed) {
    Edge e;
    e.parent = a["parent_link"].as<std::string>();
    const auto xyz = seq_or(a["origin_xyz"], {0, 0, 0});
    const auto rpy = seq_or(a["origin_rpy"], {0, 0, 0});
    e.xyzrpy = {xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]};
    e.axis = {0, 0, 1};
    edge_of[a["child_link"].as<std::string>()] = e;
  }
  for (const auto& m : robot["mimic_joints"]) {
    Edge e;
    e.parent = m["parent_link"].as<std::string>();
    e.kind = m["joint_type"].as<std::string>() == "prismatic" ? 4 : 3;
    e.dof = column.at(m["mimics"].as<std::string>());
    mimicked.push_back(static_cast<int>(e.dof));
    const auto xyz = seq_or(m["origin_xyz"], {0, 0, 0});
    const auto rpy = seq_or(m["origin_rpy"], {0, 0, 0});
    e.xyzrpy = {xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]};
    e.axis = seq_or(m["axis_xyz"], {0, 0, 1});
    e.mult = m["multiplier"] ? m["multiplier"].as<double>() : 1.0;
    e.off = m["offset"] ? m["offset"].as<double>() : 0.0;
    edge_of[m["child_link"].as<std::string>()] = e;
  }
  std::map<std::string, std::vector<std::string>> children;  // std::map: name order
  std::vector<std::string> roots;
  for (const auto& [child, e] : edge_of) {
    children[e.parent].push_back(child);
  }
  for (const auto& [parent, kids] : children) {
    if (edge_of.count(parent) == 0) {
      roots.push_back(parent);
    }
  }
  bridge::SelfModelParams p;
  std::map<std::string, std::int64_t> index;
  std::vector<std::string> queue = roots;
  for (std::size_t i = 0; i < queue.size(); ++i) {
    const std::string link = queue[i];
    index[link] = static_cast<std::int64_t>(p.link_names.size());
    p.link_names.push_back(link);
    const auto it = edge_of.find(link);
    const Edge e =
        it == edge_of.end() ? Edge{"", 0, -1, {0, 0, 0, 0, 0, 0}, {0, 0, 1}} : it->second;
    p.parent.push_back(it == edge_of.end() ? -1 : index.at(e.parent));
    p.joint_kind.push_back(e.kind);
    p.dof_index.push_back(e.dof);
    p.origin_xyzrpy.insert(p.origin_xyzrpy.end(), e.xyzrpy.begin(), e.xyzrpy.end());
    p.axis.insert(p.axis.end(), e.axis.begin(), e.axis.end());
    p.mimic_multiplier.push_back(e.mult);
    p.mimic_offset.push_back(e.off);
    for (const auto& kid : children[link]) {
      queue.push_back(kid);
    }
  }
  p.n_links = static_cast<std::int64_t>(p.link_names.size());
  return p;
}

}  // namespace

TEST(SelfFilterFk, EveryManifestWithMimicJointsMatchesTheKernelOverTheGripperStroke) {
  const auto robots = repo_root() / "robots";
  if (!std::filesystem::exists(robots)) {
    GTEST_SKIP() << "no " << robots << " (not a source checkout)";
  }
  int manifests = 0;
  for (const auto& dir : std::filesystem::directory_iterator(robots)) {
    const auto path = dir.path() / "robot.yaml";
    if (!std::filesystem::exists(path)) {
      continue;
    }
    const YAML::Node robot = YAML::LoadFile(path.string());
    if (!robot["mimic_joints"] || robot["mimic_joints"].size() == 0) {
      continue;
    }
    ++manifests;
    std::vector<std::pair<double, double>> limits;
    std::vector<int> mimicked;
    const auto params = manifest_chain(robot, limits, mimicked);
    bridge::SelfModel model;
    std::string error;
    ASSERT_TRUE(bridge::build_self_model(params, model, error)) << path << ": " << error;
    const auto kernel = kernel_model(params);
    osk::CollisionScratch scratch;
    scratch.link_world.resize(kernel.n_links);
    std::vector<tf2::Transform> ours;
    std::mt19937 rng(7);
    std::vector<double> q(limits.size());
    for (int step = 0; step <= 20; ++step) {
      for (std::size_t d = 0; d < q.size(); ++d) {
        q[d] = std::uniform_real_distribution<double>(limits[d].first, limits[d].second)(rng);
      }
      for (const int d : mimicked) {  // the gripper stroke, end to end
        const auto [lo, hi] = limits[static_cast<std::size_t>(d)];
        q[static_cast<std::size_t>(d)] = lo + (hi - lo) * step / 20.0;
      }
      osk::forward_kinematics(kernel, q.data(), q.size(), scratch);
      bridge::self_forward_kinematics(model, q.data(), q.size(), ours);
      for (std::size_t i = 0; i < kernel.n_links; ++i) {
        const auto& k = scratch.link_world[i];
        const tf2::Vector3 t(k.t.x, k.t.y, k.t.z);
        EXPECT_LT((ours[i].getOrigin() - t).length(), 1e-6)
            << path << " link " << params.link_names[i] << " step " << step;
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            EXPECT_NEAR(ours[i].getBasis()[row][col], k.r[row * 3 + col], 1e-9);
          }
        }
      }
    }
  }
  EXPECT_GE(manifests, 3) << "so101_follower, openarm and droid_franka_robotiq at least";
}

// ---------------------------------------------------------------------------
// Joint states at the stamp: interpolation and refusal.
// ---------------------------------------------------------------------------

namespace {

constexpr std::int64_t kMs = 1'000'000;

std::deque<bridge::JointSample> samples(std::vector<std::pair<std::int64_t, double>> at) {
  std::deque<bridge::JointSample> h;
  for (const auto& [t, q] : at) {
    h.push_back(bridge::JointSample{t * kMs, {q}});
  }
  return h;
}

}  // namespace

TEST(SelfFilterJointState, InterpolatesBetweenTheBracketingSamples) {
  const auto h = samples({{0, 0.0}, {10, 1.0}, {20, 3.0}});
  std::vector<double> q;
  std::vector<double> qdot;
  ASSERT_EQ(bridge::joint_state_at(h, 15 * kMs, 0.01, 0.1, q, qdot),
            bridge::JointStateVerdict::kOk);
  EXPECT_NEAR(q[0], 2.0, 1e-12);
  EXPECT_NEAR(qdot[0], 200.0, 1e-9) << "rad/s over the bracketing pair";
  ASSERT_EQ(bridge::joint_state_at(h, 10 * kMs, 0.01, 0.1, q, qdot),
            bridge::JointStateVerdict::kOk);
  EXPECT_DOUBLE_EQ(q[0], 1.0) << "a stamp on a sample is that sample";
  // Extrapolation within one period, from the two newest samples.
  ASSERT_EQ(bridge::joint_state_at(h, 25 * kMs, 0.01, 0.1, q, qdot),
            bridge::JointStateVerdict::kOk);
  EXPECT_NEAR(q[0], 4.0, 1e-12);
}

TEST(SelfFilterJointState, RefusesAGapAndAnExtrapolationAndKeepsQ) {
  std::vector<double> q = {42.0};
  std::vector<double> qdot;
  const auto gap = samples({{0, 0.0}, {25, 1.0}});
  EXPECT_EQ(bridge::joint_state_at(gap, 12 * kMs, 0.01, 0.1, q, qdot),
            bridge::JointStateVerdict::kGap)
      << "samples 2.5 periods apart";
  const auto h = samples({{0, 0.0}, {10, 1.0}});
  EXPECT_EQ(bridge::joint_state_at(h, 21 * kMs, 0.01, 0.1, q, qdot),
            bridge::JointStateVerdict::kExtrapolation)
      << "1.1 periods past the newest";
  EXPECT_EQ(bridge::joint_state_at(h, -11 * kMs, 0.01, 0.1, q, qdot),
            bridge::JointStateVerdict::kExtrapolation);
  EXPECT_EQ(bridge::joint_state_at(samples({{0, 0.0}}), 1 * kMs, 0.01, 0.1, q, qdot),
            bridge::JointStateVerdict::kExtrapolation)
      << "one sample cannot be extrapolated";
  EXPECT_EQ(bridge::joint_state_at({}, 0, 0.01, 0.1, q, qdot), bridge::JointStateVerdict::kNoState);
  EXPECT_EQ(bridge::joint_state_at(samples({{0, 0.0}, {100, 1.0}}), 50 * kMs, 0.1, 0.02, q, qdot),
            bridge::JointStateVerdict::kNoState)
      << "nearest sample past max_skew_s";
  EXPECT_DOUBLE_EQ(q[0], 42.0) << "a refusal never writes a pose";
  // One sample exactly at the stamp is the pose, at rest.
  ASSERT_EQ(bridge::joint_state_at(samples({{0, 0.5}}), 0, 0.01, 0.1, q, qdot),
            bridge::JointStateVerdict::kOk);
  EXPECT_DOUBLE_EQ(q[0], 0.5);
  EXPECT_DOUBLE_EQ(qdot[0], 0.0);
}

TEST(SelfFilterVelocityPadding, GrowsWithLinkSpeedUpToTheCap) {
  const auto params = chain_params();
  bridge::SelfModel model;
  std::string error;
  ASSERT_TRUE(bridge::build_self_model(params, model, error)) << error;
  const std::vector<double> q = {0.3, 0.1, -0.2};
  std::vector<tf2::Transform> now;
  bridge::self_forward_kinematics(model, q.data(), q.size(), now);
  std::vector<double> previous(model.primitives.size(), -1.0);
  for (const double rate : {0.0, 0.5, 2.0}) {
    std::vector<double> later_q = q;
    for (auto& v : later_q) {
      v += rate * 0.02;  // δt_sync = 20 ms
    }
    std::vector<tf2::Transform> later;
    bridge::self_forward_kinematics(model, later_q.data(), later_q.size(), later);
    std::vector<double> motion;
    bridge::self_primitive_motion(model, now, later, motion);
    ASSERT_EQ(motion.size(), model.primitives.size());
    for (std::size_t k = 0; k < motion.size(); ++k) {
      EXPECT_GT(motion[k], previous[k]) << "primitive " << k << " at rate " << rate;
      // Never an under-report: every surface point moved by at most that much.
      const tf2::Transform a =
          now[static_cast<std::size_t>(model.primitive_link[k])] * model.primitives[k].pose;
      const tf2::Transform b =
          later[static_cast<std::size_t>(model.primitive_link[k])] * model.primitives[k].pose;
      const tf2::Vector3 tip(0.0, 0.0,
                             -model.primitives[k].half_length - model.primitives[k].radius);
      EXPECT_LE(((b * tip) - (a * tip)).length(), motion[k] + 1e-12);
    }
    previous = motion;
    if (rate == 0.0) {
      EXPECT_DOUBLE_EQ(motion[0], 0.0);
    }
  }
  // A point 3 cm off the sphere: inside 2 cm + 1.5 cm of motion, outside 2 cm alone.
  std::vector<bridge::PayloadPrimitive> placed;
  bridge::place_self_primitives(model, now, tf2::Transform::getIdentity(), placed);
  const tf2::Vector3 probe = placed[1].pose * tf2::Vector3(0.0, 0.0, 0.04 + 0.03);
  bridge::SelfMask still;
  still.set({placed[1]}, 0.02, {}, {0.0});
  bridge::SelfMask moving;
  moving.set({placed[1]}, 0.02, {}, {0.015});
  EXPECT_FALSE(still.contains(probe));
  EXPECT_TRUE(moving.contains(probe));
  bridge::SelfMask fast;
  fast.set(placed, 0.02, {}, {5.0, std::nan("")});
  EXPECT_DOUBLE_EQ(fast.padding(0), bridge::kMaxSelfFilterPaddingM) << "the blind-shell cap holds";
  EXPECT_DOUBLE_EQ(fast.padding(1), 0.02) << "a non-finite speed adds nothing";
}

// ---------------------------------------------------------------------------
// Robot mask: the z-test and the uncertain band.
// ---------------------------------------------------------------------------

namespace {

constexpr std::size_t kW = 64;
constexpr std::size_t kH = 48;

bridge::PinholeCamera test_camera() {
  bridge::PinholeCamera c;
  c.fx = 200.0;
  c.fy = 200.0;
  c.cx = 32.0;
  c.cy = 24.0;
  c.width = kW;
  c.height = kH;
  return c;
}

// A 2 cm "fingertip" sphere 0.5 m ahead of the camera, on the optical axis.
bridge::PayloadPrimitive finger() {
  bridge::PayloadPrimitive s;
  s.shape_type = Wire::SHAPE_CAPSULE;
  s.radius = 0.02;
  s.pose.setOrigin(tf2::Vector3(0.0, 0.0, 0.5));
  return s;
}

std::vector<std::uint8_t> render(const std::vector<float>& depth, double margin, int k) {
  bridge::RobotMaskScratch scratch;
  std::vector<std::uint8_t> mask;
  bridge::render_robot_mask({finger()}, {nullptr}, {margin}, depth.data(), test_camera(), k,
                            scratch, mask);
  return mask;
}

std::uint8_t at(const std::vector<std::uint8_t>& m, std::size_t u, std::size_t v) {
  return m[v * kW + u];
}

}  // namespace

TEST(SelfFilterRobotMask, AFingerInFrontOfAnObjectIsRobot) {
  // The object's surface at 0.8 m: the finger (front at 0.48 m) is in front.
  const auto mask = render(std::vector<float>(kW * kH, 0.8F), 0.0, 2);
  EXPECT_EQ(at(mask, 32, 24), bridge::kRobotMaskRobot);
  // The silhouette's radius is ~8 px: 7 px out is the band, 20 px out is free.
  EXPECT_EQ(at(mask, 32 + 9, 24), bridge::kRobotMaskUncertain);
  EXPECT_EQ(at(mask, 32 + 20, 24), bridge::kRobotMaskFree);
  EXPECT_EQ(at(mask, 0, 0), bridge::kRobotMaskFree);
}

TEST(SelfFilterRobotMask, AnObjectInFrontOfAFingerIsKept) {
  // A held object's surface at 0.4 m, in front of the finger: never robot.
  const auto mask = render(std::vector<float>(kW * kH, 0.4F), bridge::kRobotMaskArmMarginM, 2);
  EXPECT_EQ(std::count(mask.begin(), mask.end(), bridge::kRobotMaskRobot), 0);
  EXPECT_EQ(std::count(mask.begin(), mask.end(), bridge::kRobotMaskUncertain), 0);
}

TEST(SelfFilterRobotMask, NoReturnOverTheRobotIsUncertainAndTheBackstopCatchesNoise) {
  std::vector<float> depth(kW * kH, 0.8F);
  depth[24 * kW + 32] = std::nanf("");
  auto mask = render(depth, 0.0, 0);
  EXPECT_EQ(at(mask, 32, 24), bridge::kRobotMaskUncertain) << "the robot covers it, no return";
  EXPECT_EQ(at(mask, 33, 24), bridge::kRobotMaskRobot);
  // A return 8 mm in front of the finger's front (0.472 m): the z-test misses it
  // (0.48 > 0.472 + 5 mm); the arm's 10 mm backstop catches it, the hand's 0 does not.
  depth.assign(kW * kH, 0.472F);
  EXPECT_EQ(at(render(depth, bridge::kRobotMaskArmMarginM, 0), 32, 24), bridge::kRobotMaskRobot);
  EXPECT_EQ(at(render(depth, bridge::kRobotMaskHandMarginM, 0), 32, 24), bridge::kRobotMaskFree);
}

TEST(SelfFilterRobotMask, RendersABoxAndAHulledBoxBySlabs) {
  bridge::PayloadPrimitive box;
  box.shape_type = Wire::SHAPE_BOX;
  box.half_extents = tf2::Vector3(0.05, 0.05, 0.05);
  box.pose.setOrigin(tf2::Vector3(0.0, 0.0, 0.5));
  bridge::SelfHull hull;  // the inscribed octahedron's DOP
  const double c = 0.05 / std::sqrt(3.0);
  const double e = 0.05 / std::sqrt(2.0);
  const double hi[13] = {0.05, 0.05, 0.05, c, c, c, c, e, e, e, e, e, e};
  for (int i = 0; i < 13; ++i) {
    hull.dop_hi[i] = hi[i];
    hull.dop_lo[i] = -hi[i];
  }
  const std::vector<float> depth(kW * kH, 1.0F);
  bridge::RobotMaskScratch scratch;
  std::vector<std::uint8_t> plain;
  std::vector<std::uint8_t> hulled;
  bridge::render_robot_mask({box}, {nullptr}, {0.0}, depth.data(), test_camera(), 0, scratch,
                            plain);
  bridge::render_robot_mask({box}, {&hull}, {0.0}, depth.data(), test_camera(), 0, scratch, hulled);
  // The box's corner direction (u, v) = (32 + 18, 24 + 18) is box, not octahedron.
  EXPECT_EQ(at(plain, 32, 24), bridge::kRobotMaskRobot);
  EXPECT_EQ(at(hulled, 32, 24), bridge::kRobotMaskRobot);
  EXPECT_EQ(at(plain, 50, 42), bridge::kRobotMaskRobot);
  EXPECT_EQ(at(hulled, 50, 42), bridge::kRobotMaskFree);
}

// ---------------------------------------------------------------------------
// Real data (issue #418): the sf_dr2 frame that seeded a ghost voxel.
// ---------------------------------------------------------------------------

namespace {

// The manifest's whole collision model (chain + every box and its tight geometry),
// the arrays the deploy launch hands the kernel and this filter.
bridge::SelfModelParams manifest_model(const YAML::Node& robot) {
  std::vector<std::pair<double, double>> limits;
  std::vector<int> mimicked;
  auto p = manifest_chain(robot, limits, mimicked);
  for (const auto& geom : robot["collision_geometry"]) {
    const auto name = geom["link_name"].as<std::string>();
    const auto it = std::find(p.link_names.begin(), p.link_names.end(), name);
    if (it == p.link_names.end() || geom["shape"]["shape"].as<std::string>() != "box") {
      continue;
    }
    p.box_link.push_back(it - p.link_names.begin());
    const auto he = geom["shape"]["half_extents_m"].as<std::vector<double>>();
    p.box_half_extents.insert(p.box_half_extents.end(), he.begin(), he.end());
    const auto o = geom["origin_xyz_rpy"].as<std::vector<double>>();
    p.box_origin_xyzrpy.insert(p.box_origin_xyzrpy.end(), o.begin(), o.end());
    const auto tight = geom["tight_geometry"];
    if (!tight) {
      p.box_hull.push_back(-1);
      continue;
    }
    p.box_hull.push_back(static_cast<std::int64_t>(p.hull_vertex_first.size()));
    const auto lo = tight["dop_lo_m"].as<std::vector<double>>();
    const auto hi = tight["dop_hi_m"].as<std::vector<double>>();
    p.hull_dop_lo.insert(p.hull_dop_lo.end(), lo.begin(), lo.end());
    p.hull_dop_hi.insert(p.hull_dop_hi.end(), hi.begin(), hi.end());
    p.hull_vertex_first.push_back(static_cast<std::int64_t>(p.hull_vertices.size() / 3));
    for (const auto& v : tight["hull_vertices_m"]) {
      const auto xyz = v.as<std::vector<double>>();
      p.hull_vertices.insert(p.hull_vertices.end(), xyz.begin(), xyz.end());
    }
    p.hull_vertex_count.push_back(static_cast<std::int64_t>(p.hull_vertices.size() / 3) -
                                  p.hull_vertex_first.back());
  }
  return p;
}

std::vector<tf2::Vector3> points_of(const YAML::Node& n) {
  std::vector<tf2::Vector3> out;
  for (const auto& p : n) {
    const auto v = p.as<std::vector<double>>();
    out.emplace_back(v[0], v[1], v[2]);
  }
  return out;
}

// The node's mask for one cloud: joint states interpolated at the stamp, padding
// grown by each primitive's motion over `velocity_padding_dt_s` (0.02 s), as
// `RobotSelfFilter::on_cloud` builds it.
bool real_frame_mask(const bridge::SelfModel& model, const YAML::Node& fx,
                     const YAML::Node& history, bridge::SelfMask& mask) {
  std::deque<bridge::JointSample> h;
  for (const auto& s : history) {  // inserted by stamp, as `on_joint_state` does
    bridge::JointSample sample{s["stamp_ns"].as<std::int64_t>(), s["q"].as<std::vector<double>>()};
    const auto at = std::upper_bound(
        h.begin(), h.end(), sample.stamp_ns,
        [](std::int64_t t, const bridge::JointSample& x) { return t < x.stamp_ns; });
    h.insert(at, sample);
  }
  std::vector<double> q;
  std::vector<double> qdot;
  if (bridge::joint_state_at(h, fx["cloud_stamp_ns"].as<std::int64_t>(),
                             fx["joint_state_period_s"].as<double>(), 0.1, q,
                             qdot) != bridge::JointStateVerdict::kOk) {
    return false;
  }
  std::vector<double> later(q.size());
  for (std::size_t d = 0; d < q.size(); ++d) {
    later[d] = q[d] + qdot[d] * 0.02;
  }
  std::vector<tf2::Transform> now_w;
  std::vector<tf2::Transform> later_w;
  bridge::self_forward_kinematics(model, q.data(), q.size(), now_w);
  bridge::self_forward_kinematics(model, later.data(), later.size(), later_w);
  std::vector<double> motion;
  bridge::self_primitive_motion(model, now_w, later_w, motion);
  const auto t = fx["cloud_from_root"]["xyz"].as<std::vector<double>>();
  const auto r = fx["cloud_from_root"]["quat_xyzw"].as<std::vector<double>>();
  const tf2::Transform cloud_from_root(tf2::Quaternion(r[0], r[1], r[2], r[3]),
                                       tf2::Vector3(t[0], t[1], t[2]));
  std::vector<bridge::PayloadPrimitive> prims;
  std::vector<const bridge::SelfHull*> hulls;
  bridge::place_self_primitives(model, now_w, cloud_from_root, prims, &hulls);
  mask.set(prims, 0.02, hulls, motion);
  return true;
}

std::size_t kept(const bridge::SelfMask& mask, const std::vector<tf2::Vector3>& pts) {
  return static_cast<std::size_t>(std::count_if(
      pts.begin(), pts.end(), [&](const tf2::Vector3& p) { return !mask.contains(p); }));
}

}  // namespace

TEST(SelfFilterRealData, SimTimeStampedJointStatesRemoveTheArmTheRecordedStampsLeft) {
  // sf_dr2 (Isaac 6.1 deploy sim, ~0.13x real time): the HAL stamped each
  // republish of a post-step frame "node now minus its WALL age", so one pose
  // carried stamps reaching seconds into the past. At this cloud's stamp the
  // filter interpolated a pose 15-25 mm / up to 10 deg from the one the camera
  // saw, and kept 30 returns lying on the robot (left_bar 14, left_distal_phalanx
  // 12, left_outer_knuckle 4); two of them became voxel 1352907, which stopped
  // panda_link7 against itself. With each frame stamped at its sim time (the
  // HAL fix) the same cloud loses every robot return and keeps the world.
  const auto manifest = repo_root() / "robots" / "droid_franka_robotiq" / "robot.yaml";
  const auto fixture = repo_root() / "packages" / "openral_octomap_bridge" / "test" / "data" /
                       "sf_dr2_ghost_frame.yaml";
  if (!std::filesystem::exists(manifest) || !std::filesystem::exists(fixture)) {
    GTEST_SKIP() << "no " << manifest << " / " << fixture << " (not a source checkout)";
  }
  const YAML::Node robot = YAML::LoadFile(manifest.string());
  const YAML::Node fx = YAML::LoadFile(fixture.string());
  bridge::SelfModel model;
  std::string error;
  ASSERT_TRUE(bridge::build_self_model(manifest_model(robot), model, error)) << error;
  ASSERT_EQ(model.primitives.size(), 17U);
  const auto names = fx["joint_names"].as<std::vector<std::string>>();
  for (std::size_t d = 0; d < names.size(); ++d) {
    ASSERT_EQ(robot["joints"][d]["name"].as<std::string>(), names[d]) << "dof column " << d;
  }
  const auto leaked = points_of(fx["leaked_points"]);
  const auto robot_pts = points_of(fx["robot_points"]);
  const auto world_pts = points_of(fx["world_points"]);
  ASSERT_EQ(leaked.size(), 30U);

  bridge::SelfMask recorded;
  ASSERT_TRUE(real_frame_mask(model, fx, fx["recorded_history"], recorded));
  const std::size_t recorded_kept = kept(recorded, leaked);
  RecordProperty("robot_returns_kept_recorded_stamps", static_cast<int>(recorded_kept));
  EXPECT_GT(recorded_kept, 0U) << "the recorded stamps pose the hand off its returns";

  bridge::SelfMask sim_time;
  ASSERT_TRUE(real_frame_mask(model, fx, fx["sim_time_history"], sim_time));
  EXPECT_EQ(kept(sim_time, leaked), 0U) << "a robot return survived the sim-time pose";
  EXPECT_EQ(kept(sim_time, robot_pts), 0U) << "of " << robot_pts.size() << " arm returns";
  EXPECT_EQ(kept(sim_time, world_pts), world_pts.size())
      << "a world return 4-8 cm from the robot was removed";
}

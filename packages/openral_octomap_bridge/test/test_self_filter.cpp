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
#include <filesystem>
#include <limits>
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

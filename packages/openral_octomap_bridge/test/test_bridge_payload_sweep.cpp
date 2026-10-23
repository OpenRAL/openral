// SPDX-License-Identifier: Apache-2.0
// Geometry-only attachment evidence cannot authorize world occupancy removal.
// Only an identity-validated tracked-object binding can clear cells.
//
// Runs the REAL node in-process against real TF (/tf history) and real DDS, with a real
// `BoundingBoxQuery` server standing in for `octomap_server`'s `~/clear_bbox` (a process
// boundary): every request the bridge sends is recorded.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gtest/gtest.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/srv/bounding_box_query.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>

#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/attached_collision_primitive.hpp>
#include <openral_msgs/msg/occupancy_voxels.hpp>
#include <openral_msgs/msg/world_state_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "octomap_voxel_bridge.hpp"

namespace bridge = openral_octomap_bridge;
using namespace std::chrono_literals;
using BBox = octomap_msgs::srv::BoundingBoxQuery;

namespace {

constexpr double kRes = 0.015;
constexpr double kHalf[3] = {0.032, 0.033, 0.054};  // g12's accepted payload
constexpr double kSupportTop = 0.030;               // the counter
constexpr double kLineY = 0.0075;
constexpr double kLineZ = 0.0975;
constexpr double kHalfId = 0.02;
tf2::Vector3 line_cell(int i, double z = kLineZ) { return {(i + 0.5) * kRes, kLineY, z}; }

class BridgePayloadSweep : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    std::vector<rclcpp::Parameter> overrides{
        {"base_frame", "base_link"},
        {"octomap_topic", "/test_octomap_sweep"},
        {"output_topic", "/test_world_voxels_sweep"},
        {"world_state_topic", "/test_world_state_sweep"},
        {"resolution", kRes},
        {"coverage_radius_m", 0.4},
        {"publish_rate_hz", 20.0},
        {"payload_sweep_clear_service", "/test_clear_bbox"},
        {"payload_sweep_clear_period_s", 0.2},
        {"static_points_file", static_points_file_},
    };
    rclcpp::NodeOptions options;
    options.parameter_overrides(overrides);
    bridge_ = std::make_shared<bridge::OctomapVoxelBridge>(options);
    peer_ = std::make_shared<rclcpp::Node>("bridge_payload_sweep_peer");
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(peer_);
    octomap_pub_ = peer_->create_publisher<octomap_msgs::msg::Octomap>("/test_octomap_sweep",
                                                                       rclcpp::QoS(1).reliable());
    world_state_pub_ = peer_->create_publisher<openral_msgs::msg::WorldStateStamped>(
        "/test_world_state_sweep", rclcpp::QoS(1).reliable());
    grid_sub_ = peer_->create_subscription<openral_msgs::msg::OccupancyVoxels>(
        "/test_world_voxels_sweep", rclcpp::QoS(10).reliable(),
        [this](openral_msgs::msg::OccupancyVoxels::ConstSharedPtr g) {
          const std::lock_guard<std::mutex> lock(mutex_);
          grid_revisions_.push_back(g->attachment_revision);
          last_grid_ = g;
        });
    server_ = peer_->create_service<BBox>(
        "/test_clear_bbox",
        [this](const std::shared_ptr<BBox::Request> req, std::shared_ptr<BBox::Response> /*resp*/) {
          const std::lock_guard<std::mutex> lock(mutex_);
          requests_.push_back(*req);
        });
    executor_.add_node(bridge_);
    executor_.add_node(peer_);
    spin_for(300ms);  // discovery
  }

  void TearDown() override {
    executor_.remove_node(peer_);
    executor_.remove_node(bridge_);
  }

  void spin_for(std::chrono::milliseconds d) {
    const auto end = std::chrono::steady_clock::now() + d;
    while (std::chrono::steady_clock::now() < end) {
      executor_.spin_some(10ms);
      std::this_thread::sleep_for(2ms);
    }
  }

  // The hand (= payload centre) at x along the carry, on /tf with the current stamp.
  void place_hand(double x, double z) {
    geometry_msgs::msg::TransformStamped base;
    base.header.stamp = peer_->now();
    base.header.frame_id = "map";
    base.child_frame_id = "base_link";
    base.transform.rotation.w = 1.0;
    geometry_msgs::msg::TransformStamped hand = base;
    hand.header.frame_id = "base_link";
    hand.child_frame_id = "hand";
    hand.transform.translation.x = x;
    hand.transform.translation.z = z;
    tf_->sendTransform({base, hand});
  }

  void send_octree(const std::vector<tf2::Vector3>& cells = {}) {
    octomap::OcTree tree(kRes);
    tree.updateNode(octomap::point3d(0.3F, 0.3F, 0.3F), true);  // anything: a live octree
    for (const auto& c : cells) {
      tree.updateNode(octomap::point3d(static_cast<float>(c.x()), static_cast<float>(c.y()),
                                       static_cast<float>(c.z())),
                      true);
    }
    octomap_msgs::msg::Octomap msg;
    ASSERT_TRUE(octomap_msgs::binaryMapToMsg(tree, msg));
    msg.header.frame_id = "map";
    msg.header.stamp = peer_->now();
    octomap_pub_->publish(msg);
  }

  void send_world_state(bool attest_support = true,
                        const std::string& evidence_kind = "vision_segmentation") {
    openral_msgs::msg::AttachedCollisionObject object;
    object.object_id = "grasped_payload:gripper";
    object.attach_link = "hand";
    object.pose_in_link.orientation.w = 1.0;
    openral_msgs::msg::AttachedCollisionPrimitive box;
    box.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_BOX;
    box.shape_dimensions = {kHalf[0], kHalf[1], kHalf[2]};
    box.pose_in_object.orientation.w = 1.0;
    object.primitives.push_back(box);
    object.evidence_kind = evidence_kind;
    object.support_contact_valid = attest_support;
    auto& w = object.support_contact;
    w.support_id = "counter";
    w.contact_point_in_object.z = -kHalf[2];
    w.contact_normal_in_object.z = 1.0;
    w.patch_radius_m = 0.03;
    w.max_penetration_m = 0.002;
    w.stamp_ns = 1;
    openral_msgs::msg::WorldStateStamped state;
    state.header.stamp = peer_->now();
    state.attached_objects.push_back(object);
    state.attachment_revision = 7;
    world_state_pub_->publish(state);
  }

  /// How many clear_bbox requests contain `p`.
  int times_cleared(const tf2::Vector3& p) {
    const std::lock_guard<std::mutex> lock(mutex_);
    int n = 0;
    for (const auto& r : requests_) {
      n += (r.min.x <= p.x() && p.x() <= r.max.x && r.min.y <= p.y() && p.y() <= r.max.y &&
            r.min.z <= p.z() && p.z() <= r.max.z)
               ? 1
               : 0;
    }
    return n;
  }

  /// The newest published grid's `attachment_revision` (0 if none yet).
  std::uint64_t last_grid_revision() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return grid_revisions_.empty() ? 0 : grid_revisions_.back();
  }

  /// The highest `attachment_revision` any grid has carried.
  std::uint64_t max_grid_revision() {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::uint64_t m = 0;
    for (const auto r : grid_revisions_) {
      m = std::max(m, r);
    }
    return m;
  }

  bool last_grid_cell_occupied(const tf2::Vector3& point) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!last_grid_ || !(last_grid_->resolution > 0.0)) {
      return false;
    }
    const tf2::Quaternion orientation(last_grid_->orientation.x, last_grid_->orientation.y,
                                      last_grid_->orientation.z, last_grid_->orientation.w);
    const tf2::Transform frame_from_grid(
        orientation,
        tf2::Vector3(last_grid_->origin.x, last_grid_->origin.y, last_grid_->origin.z));
    const tf2::Vector3 local = frame_from_grid.inverse() * point;
    const long ix = static_cast<long>(std::floor(local.x() / last_grid_->resolution));
    const long iy = static_cast<long>(std::floor(local.y() / last_grid_->resolution));
    const long iz = static_cast<long>(std::floor(local.z() / last_grid_->resolution));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= static_cast<long>(last_grid_->size_x) ||
        iy >= static_cast<long>(last_grid_->size_y) ||
        iz >= static_cast<long>(last_grid_->size_z)) {
      return false;
    }
    const auto index =
        static_cast<std::size_t>(ix) +
        static_cast<std::size_t>(last_grid_->size_x) *
            (static_cast<std::size_t>(iy) +
             static_cast<std::size_t>(last_grid_->size_y) * static_cast<std::size_t>(iz));
    return index < last_grid_->occupancy.size() && last_grid_->occupancy[index] != 0;
  }

  void expect_unowned_geometry_stays_mapped(const std::string& evidence_kind) {
    const auto cell = line_cell(6);
    for (int i = 0; i < 8; ++i) {
      place_hand(0.09, kLineZ);
      send_octree({cell});
      send_world_state(false, evidence_kind);
      spin_for(50ms);
    }
    spin_for(300ms);
    EXPECT_TRUE(last_grid_cell_occupied(cell)) << evidence_kind << " cleared an unowned cell";
    EXPECT_TRUE(requests_.empty()) << evidence_kind << " issued a raw clear_bbox request";
    EXPECT_EQ(last_grid_revision(), 7U)
        << "processing the attachment revision should still certify";
  }

  std::string static_points_file_;
  rclcpp::Subscription<openral_msgs::msg::OccupancyVoxels>::SharedPtr grid_sub_;
  std::vector<std::uint64_t> grid_revisions_;
  openral_msgs::msg::OccupancyVoxels::ConstSharedPtr last_grid_;
  std::shared_ptr<bridge::OctomapVoxelBridge> bridge_;
  std::shared_ptr<rclcpp::Node> peer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_pub_;
  rclcpp::Publisher<openral_msgs::msg::WorldStateStamped>::SharedPtr world_state_pub_;
  rclcpp::Service<BBox>::SharedPtr server_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::mutex mutex_;
  std::vector<BBox::Request> requests_;
};

TEST_F(BridgePayloadSweep, RegionEvidenceDoesNotAuthorizeGridOrOctreeClearing) {
  expect_unowned_geometry_stays_mapped("grasp_target_region");
}

TEST_F(BridgePayloadSweep, SegmentationEvidenceDoesNotAuthorizeGridOrOctreeClearing) {
  expect_unowned_geometry_stays_mapped("vision_segmentation");
}

TEST_F(BridgePayloadSweep, UnknownEvidenceDoesNotAuthorizeGridOrOctreeClearing) {
  expect_unowned_geometry_stays_mapped("unrecognized_evidence");
}

TEST_F(BridgePayloadSweep, PinchFallbackDoesNotAuthorizeGridOrOctreeClearing) {
  expect_unowned_geometry_stays_mapped("pinch_fallback");
}

TEST_F(BridgePayloadSweep, GripperClosureDoesNotAuthorizeGridOrOctreeClearing) {
  expect_unowned_geometry_stays_mapped("gripper_closure");
}

TEST_F(BridgePayloadSweep, NothingIsClearedWithoutAnAttachment) {
  for (int i = 0; i <= 10; ++i) {
    place_hand(0.01 * i, 0.1);
    send_octree();
    spin_for(50ms);
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  EXPECT_TRUE(requests_.empty());
}

// ── Clearing by identity (issue #418) ───────────────────────────────────────

openral_msgs::msg::TrackedObject tracked(const std::string& id, const std::string& status, double x,
                                         double half, double floor_z) {
  openral_msgs::msg::TrackedObject t;
  t.object_id = id;
  t.status = status;
  t.box.frame_id = "base_link";
  t.box.pose.position.x = x;
  t.box.pose.position.y = kLineY;
  t.box.pose.position.z = kLineZ;
  t.box.pose.orientation.w = 1.0;
  t.box.half_extents.x = t.box.half_extents.y = t.box.half_extents.z = half;
  t.floor_z = floor_z;
  if (status == "bound") {
    t.attach_link = "hand";
  }
  return t;
}

class BridgeIdentitySweep : public BridgePayloadSweep {
protected:
  void SetUp() override {
    // One declared static point on the carry line (cell 3).
    static_points_file_ = std::string(::testing::TempDir()) + "identity_static_points.txt";
    std::ofstream(static_points_file_)
        << line_cell(3).x() << " " << kLineY << " " << kLineZ << "\n";
    BridgePayloadSweep::SetUp();
  }
  void TearDown() override {
    BridgePayloadSweep::TearDown();
    std::remove(static_points_file_.c_str());
  }

  void send_tracked(const std::vector<openral_msgs::msg::TrackedObject>& tracks,
                    const std::string& evidence = "", std::int64_t stamp_ns = 0, double x = 0.0) {
    openral_msgs::msg::WorldStateStamped state;
    state.header.stamp = peer_->now();
    state.attachment_revision = revision_;
    state.tracked_objects_valid = tracked_valid_;
    state.tracked_objects.frame_id = "base_link";
    for (const auto& t : tracks) {
      state.tracked_objects.objects.push_back(t);
    }
    if (!evidence.empty()) {
      openral_msgs::msg::AttachedCollisionObject object;
      object.object_id = "trk:1";
      object.attach_link = "hand";
      object.evidence_kind = evidence;
      object.stamp_ns = stamp_ns;
      object.pose_in_link.orientation.w = 1.0;
      openral_msgs::msg::AttachedCollisionPrimitive box;
      box.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_BOX;
      box.shape_dimensions = {kHalfId, kHalfId, kHalfId};
      box.pose_in_object.orientation.w = 1.0;
      object.primitives.push_back(box);
      state.attached_objects.push_back(object);
      if (released_status_.empty()) {
        state.tracked_objects.objects.push_back(tracked("trk:1", "bound", x, kHalfId, 0.09));
      } else {
        auto t = tracked("trk:1", released_status_, line_cell(18).x(), kHalfId, 0.09);
        t.epoch = released_epoch_;
        state.tracked_objects.objects.push_back(t);
      }
    }
    world_state_pub_->publish(state);
  }

  // Bound at x = 0.09, then `status` (epoch `epoch`) with its box on cells 17-19 and the
  // attachment still on the wire, `ticks` world states each.
  void release_phase(const std::string& evidence, const std::string& status, std::uint32_t epoch,
                     int ticks) {
    released_status_ = status;
    released_epoch_ = epoch;
    for (int i = 0; i < ticks; ++i) {
      place_hand(0.09, kLineZ);
      send_octree(release_cells_);
      send_tracked({}, evidence, 0, 0.09);
      spin_for(50ms);
    }
    spin_for(100ms);
  }

  std::uint64_t revision_{0};
  bool tracked_valid_{true};
  std::string released_status_;
  std::uint32_t released_epoch_{0};
  const std::vector<tf2::Vector3> release_cells_{line_cell(17), line_cell(18), line_cell(19),
                                                 line_cell(18, 0.0825)};

  // The carry line's cells, one under the 0.09 floor, run along x 0 -> 0.16; the binding
  // appears at x = 0.09 with t_attach stamped at x = 0.04.
  void carry(const std::string& evidence) {
    std::vector<tf2::Vector3> cells;
    for (int i = 0; i < 12; ++i) {
      cells.push_back(line_cell(i));
    }
    cells.push_back(line_cell(6, 0.0825));  // below floor_z
    // A packed neighbour on the line at cell 9.
    const auto neighbour = tracked("trk:2", "confirmed", line_cell(9).x(), 0.005, 0.0);
    std::int64_t t_attach = 0;
    for (int i = 0; i <= 16; ++i) {
      place_hand(0.01 * i, kLineZ);
      send_octree(cells);
      if (i == 4) {
        t_attach = peer_->now().nanoseconds();
      }
      send_tracked({neighbour}, i >= 9 ? evidence : "", t_attach, 0.01 * i);
      spin_for(50ms);
    }
    spin_for(300ms);
  }
};

TEST_F(BridgeIdentitySweep, TheCarryIsBackfilledFromAttachAndKeepsEveryExclusion) {
  carry("tracked_object");
  EXPECT_GT(times_cleared(line_cell(11)), 0) << "the live carry was not cleared";
  EXPECT_GT(times_cleared(line_cell(2)), 0) << "no backfill from t_attach";
  EXPECT_EQ(times_cleared(line_cell(0)), 0) << "cleared before t_attach";
  EXPECT_EQ(times_cleared(line_cell(6, 0.0825)), 0) << "cleared below floor_z";
  EXPECT_EQ(times_cleared(line_cell(9)), 0) << "cleared inside another track's box";
  EXPECT_EQ(times_cleared(line_cell(3)), 0) << "cleared declared static geometry";
}

// ── The grid says which attachment revision it was cleared for (#418 r7po_on1) ──────────
//
// The sim HAL's perception barrier releases the runner on a grid that carries the payload's
// clearing; r7po_on1's kernel stopped on the held orange's own cells because the barrier
// released on a grid the bridge had built BEFORE it consumed the binding.

TEST_F(BridgeIdentitySweep, AGridIsCertifiedForARevisionOnlyOnceItsBindingIsCleared) {
  revision_ = 11;
  // No attachment and no world state yet: nothing to certify.
  for (int i = 0; i < 4; ++i) {
    place_hand(0.0, kLineZ);
    send_octree({line_cell(0), line_cell(1)});
    spin_for(50ms);
  }
  EXPECT_EQ(max_grid_revision(), 0U) << "a grid claimed a revision no state had named";
  carry("tracked_object");
  EXPECT_EQ(last_grid_revision(), 11U) << "the grid built with the binding's clearing";
}

TEST_F(BridgeIdentitySweep, AnUnusableTrackedSetWithholdsTheCertificate) {
  revision_ = 12;
  tracked_valid_ = false;   // the bound payload's tracker set is unusable: nothing is cleared
  carry("tracked_object");  // the binding is on the wire from the 10th world state on
  EXPECT_EQ(last_grid_revision(), 0U) << "certified a clear that did not happen";
}

TEST_F(BridgeIdentitySweep, ALeftInPlaceBindingStillCertifiesTheRevision) {
  revision_ = 13;
  carry("pinch_fallback");  // stays an obstacle on purpose: nothing for a waiter to wait for
  EXPECT_EQ(last_grid_revision(), 13U);
}

TEST_F(BridgeIdentitySweep, APinchFallbackClearsNothing) {
  carry("pinch_fallback");
  const std::lock_guard<std::mutex> lock(mutex_);
  EXPECT_TRUE(requests_.empty());
}

TEST_F(BridgeIdentitySweep, AVacatedBoxIsClearedOnceAndNeverAgain) {
  // trk:5 confirmed at A (cells 13-15), then confirmed at B, far away. A's cells stay
  // occupied in every later octree: re-observed, so they must not be cleared again.
  const std::vector<tf2::Vector3> home{line_cell(13), line_cell(14), line_cell(15)};
  const double a = line_cell(14).x();
  for (int i = 0; i < 20; ++i) {
    place_hand(0.0, 0.3);
    send_octree(home);
    send_tracked({tracked("trk:5", "confirmed", i < 5 ? a : -0.15, 0.02, 0.05)});
    spin_for(50ms);
  }
  EXPECT_EQ(times_cleared(line_cell(14)), 1);
  EXPECT_EQ(times_cleared(line_cell(13)), 1);
}

TEST_F(BridgeIdentitySweep, AReleasedBoxIsClearedOnlyThroughItsReleaseWindow) {
  release_phase("tracked_object", "", 0, 6);
  EXPECT_EQ(times_cleared(line_cell(18)), 0) << "cleared before the binding ended";
  release_phase("tracked_object", "released", 0, 8);
  const int during = times_cleared(line_cell(18));
  EXPECT_GT(during, 0) << "the released box was not cleared in its window";
  EXPECT_GT(times_cleared(line_cell(17)), 0);
  EXPECT_EQ(times_cleared(line_cell(18, 0.0825)), 0) << "cleared below floor_z";
  release_phase("tracked_object", "confirmed", 1, 8);
  EXPECT_EQ(times_cleared(line_cell(18)), during) << "cleared after re-confirmation";
  release_phase("tracked_object", "released", 1, 8);
  EXPECT_EQ(times_cleared(line_cell(18)), during) << "reopened without a binding";
}

TEST_F(BridgeIdentitySweep, APinchFallbackReleaseClearsNothing) {
  release_phase("pinch_fallback", "", 0, 6);
  release_phase("pinch_fallback", "released", 0, 8);
  const std::lock_guard<std::mutex> lock(mutex_);
  EXPECT_TRUE(requests_.empty());
}

}  // namespace

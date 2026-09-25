// SPDX-License-Identifier: Apache-2.0
// The self-filter node, over real DDS: it forwards a cloud only when it knows
// where the robot was at the cloud's capture stamp, removes the robot's points
// when it does, and keeps the capture stamp on what it forwards.

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gtest/gtest.h>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <rclcpp/rclcpp.hpp>

#include "robot_self_filter.hpp"

using namespace std::chrono_literals;
namespace bridge = openral_octomap_bridge;

namespace {

sensor_msgs::msg::PointCloud2 make_cloud(const std::vector<std::array<float, 3>>& pts,
                                         const rclcpp::Time& stamp) {
  sensor_msgs::msg::PointCloud2 c;
  c.header.frame_id = "cam";
  c.header.stamp = stamp;
  c.height = 1;
  c.width = static_cast<std::uint32_t>(pts.size());
  c.point_step = 12;
  c.row_step = c.point_step * c.width;
  for (const auto& [name, off] :
       std::vector<std::pair<std::string, std::uint32_t>>{{"x", 0}, {"y", 4}, {"z", 8}}) {
    sensor_msgs::msg::PointField f;
    f.name = name;
    f.offset = off;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;
    c.fields.push_back(f);
  }
  c.data.resize(pts.size() * 12);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    std::memcpy(c.data.data() + i * 12, pts[i].data(), 12);
  }
  return c;
}

class SelfFilterNode : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    const std::string suffix = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    in_ = "/test_sf_in_" + suffix;
    out_ = "/test_sf_out_" + suffix;
    js_ = "/test_sf_js_" + suffix;
    rclcpp::NodeOptions options;
    options.arguments({"--ros-args", "-r", "cloud_in:=" + in_, "-r", "cloud_out:=" + out_});
    // root (fixed) -> l1 (revolute about z); the primitives ride l1.
    std::vector<rclcpp::Parameter> params = {
        {"collision_n_links", std::int64_t{2}},
        {"collision_parent", std::vector<std::int64_t>{-1, 0}},
        {"collision_joint_kind", std::vector<std::int64_t>{0, 1}},
        {"collision_dof_index", std::vector<std::int64_t>{-1, 0}},
        {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"collision_axis", std::vector<double>{0, 0, 1, 0, 0, 1}},
        {"collision_link_names", std::vector<std::string>{"root", "l1"}},
        {"collision_joint_names", std::vector<std::string>{"j1"}},
        {"joint_states_topic", js_},
        {"padding_m", 0.02},
        {"max_joint_state_skew_s", 0.1},
    };
    for (auto& p : primitive_params()) {
      params.push_back(std::move(p));
    }
    options.parameter_overrides(params);
    filter_ = std::make_shared<bridge::RobotSelfFilter>(options);
    peer_ = std::make_shared<rclcpp::Node>("self_filter_peer_" + suffix);
    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(peer_);
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = peer_->now();
    tf.header.frame_id = "cam";
    tf.child_frame_id = "root";
    tf.transform.rotation.w = 1.0;
    tf_->sendTransform(tf);
    cloud_pub_ = peer_->create_publisher<sensor_msgs::msg::PointCloud2>(
        in_, rclcpp::SensorDataQoS().keep_last(2));
    js_pub_ = peer_->create_publisher<sensor_msgs::msg::JointState>(js_, rclcpp::QoS(10));
    out_sub_ = peer_->create_subscription<sensor_msgs::msg::PointCloud2>(
        out_, rclcpp::QoS(2).reliable(), [this](sensor_msgs::msg::PointCloud2::SharedPtr m) {
          last_ = *m;
          ++received_;
        });
    executor_.add_node(filter_);
    executor_.add_node(peer_);
    spin_for(400ms);
  }

  /// One capsule on l1 at x = 0.5 m.
  virtual std::vector<rclcpp::Parameter> primitive_params() const {
    return {
        {"collision_capsule_link", std::vector<std::int64_t>{1}},
        {"collision_capsule_radius", std::vector<double>{0.05}},
        {"collision_capsule_half_length", std::vector<double>{0.05}},
        {"collision_capsule_origin_xyzrpy", std::vector<double>{0.5, 0, 0, 0, 0, 0}},
    };
  }

  void TearDown() override {
    executor_.remove_node(peer_);
    executor_.remove_node(filter_);
  }

  void spin_for(std::chrono::milliseconds d) {
    const auto end = std::chrono::steady_clock::now() + d;
    while (std::chrono::steady_clock::now() < end) {
      executor_.spin_some(10ms);
      std::this_thread::sleep_for(2ms);
    }
  }

  void send_joint_state(double q, const rclcpp::Time& stamp) {
    sensor_msgs::msg::JointState js;
    js.header.stamp = stamp;
    js.name = {"j1"};
    js.position = {q};
    js_pub_->publish(js);
  }

  std::string in_, out_, js_;
  std::shared_ptr<bridge::RobotSelfFilter> filter_;
  std::shared_ptr<rclcpp::Node> peer_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr out_sub_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  sensor_msgs::msg::PointCloud2 last_;
  std::atomic<int> received_{0};
};

}  // namespace

TEST_F(SelfFilterNode, WithoutAJointStateTheCloudIsDroppedNotForwarded) {
  cloud_pub_->publish(make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, peer_->now()));
  spin_for(400ms);
  EXPECT_EQ(received_.load(), 0) << "an unposed robot must not reach the map, filtered or not";
  EXPECT_GT(filter_->totals().dropped_no_state, 0U);
}

TEST_F(SelfFilterNode, RemovesTheRobotAtTheCloudStampAndKeepsTheWorld) {
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  // At q = 0 the capsule sits at x = 0.5; a point there is robot, one at x = 2 is world.
  cloud_pub_->publish(make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, now));
  spin_for(400ms);
  ASSERT_EQ(received_.load(), 1);
  EXPECT_EQ(last_.width, 1U) << "exactly the world point survives";
  float x = 0.0F;
  std::memcpy(&x, last_.data.data(), sizeof(float));
  EXPECT_FLOAT_EQ(x, 2.0F);
  EXPECT_EQ(rclcpp::Time(last_.header.stamp).nanoseconds(), now.nanoseconds())
      << "the capture stamp must travel on into the octree";

  // Rotate the joint by 90 degrees: the capsule is now at y = 0.5, so the point
  // at x = 0.5 is world and one at y = 0.5 is robot.
  const rclcpp::Time later = peer_->now();
  send_joint_state(M_PI / 2.0, later);
  spin_for(100ms);
  cloud_pub_->publish(make_cloud({{0.5F, 0.0F, 0.0F}, {0.0F, 0.5F, 0.0F}}, later));
  spin_for(400ms);
  ASSERT_EQ(received_.load(), 2);
  ASSERT_EQ(last_.width, 1U);
  std::memcpy(&x, last_.data.data(), sizeof(float));
  EXPECT_FLOAT_EQ(x, 0.5F) << "the filter follows the joint, not a fixed mask";
}

TEST_F(SelfFilterNode, AJointStateFarFromTheCaptureStampIsNotUsed) {
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now - rclcpp::Duration::from_seconds(1.0));
  spin_for(100ms);
  cloud_pub_->publish(make_cloud({{0.5F, 0.0F, 0.0F}}, now));
  spin_for(400ms);
  EXPECT_EQ(received_.load(), 0) << "a pose 1 s from the capture would filter the wrong place";
  EXPECT_GT(filter_->totals().dropped_no_state, 0U);
}

namespace {

// A 0.2 m cube on l1 at x = 0.5 m refined by the octahedron inscribed in it
// (vertices at +-0.1 m on each axis), in the kernel's `collision_box_hull` /
// `collision_hull_*` layout. The octahedron's 26-DOP is the octahedron itself:
// its faces are the DOP's four corner-axis slab pairs.
std::vector<rclcpp::Parameter> octahedron_box_params(std::vector<double> dop_hi_override = {}) {
  const double c = 0.1 / std::sqrt(3.0);
  const double e = 0.1 / std::sqrt(2.0);
  std::vector<double> hi = {0.1, 0.1, 0.1, c, c, c, c, e, e, e, e, e, e};
  if (!dop_hi_override.empty()) {
    hi = std::move(dop_hi_override);
  }
  std::vector<double> lo(13);
  for (std::size_t i = 0; i < 13; ++i) {
    lo[i] = -hi[i];
  }
  return {
      {"collision_box_link", std::vector<std::int64_t>{1}},
      {"collision_box_half_extents", std::vector<double>{0.1, 0.1, 0.1}},
      {"collision_box_origin_xyzrpy", std::vector<double>{0.5, 0, 0, 0, 0, 0}},
      {"collision_box_hull", std::vector<std::int64_t>{0}},
      {"collision_hull_dop_lo", lo},
      {"collision_hull_dop_hi", hi},
      {"collision_hull_vertex_first", std::vector<std::int64_t>{0}},
      {"collision_hull_vertex_count", std::vector<std::int64_t>{6}},
      {"collision_hull_vertices",
       std::vector<double>{0.1, 0, 0, -0.1, 0, 0, 0, 0.1, 0, 0, -0.1, 0, 0, 0, 0.1, 0, 0, -0.1}},
  };
}

class HulledBoxNode : public SelfFilterNode {
protected:
  std::vector<rclcpp::Parameter> primitive_params() const override {
    return octahedron_box_params();
  }
};

class MalformedHullNode : public SelfFilterNode {
protected:
  // The x slab reaches 0.15 m, outside the 0.1 m box it claims to refine.
  std::vector<rclcpp::Parameter> primitive_params() const override {
    const double c = 0.1 / std::sqrt(3.0);
    const double e = 0.1 / std::sqrt(2.0);
    return octahedron_box_params({0.15, 0.1, 0.1, c, c, c, c, e, e, e, e, e, e});
  }
};

}  // namespace

TEST_F(HulledBoxNode, FiltersAgainstTheHullNotTheBoxThatEnclosesIt) {
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  cloud_pub_->publish(make_cloud(
      {
          {0.5F, 0.0F, 0.0F},   // inside the hull: robot
          {0.61F, 0.0F, 0.0F},  // 1 cm off the hull's +x vertex, inside the 2 cm padding
          {0.6F, 0.1F, 0.1F},   // the box's corner: 11.5 cm from the hull, world
          {0.63F, 0.0F, 0.0F},  // 3 cm off the +x vertex, past the padding
      },
      now));
  spin_for(400ms);
  ASSERT_EQ(received_.load(), 1);
  ASSERT_EQ(last_.width, 2U) << "the box's corner slack is not the robot";
  float x[2];
  std::memcpy(&x[0], last_.data.data(), sizeof(float));
  std::memcpy(&x[1], last_.data.data() + 12, sizeof(float));
  EXPECT_FLOAT_EQ(x[0], 0.6F);
  EXPECT_FLOAT_EQ(x[1], 0.63F);
}

TEST_F(MalformedHullNode, AHullThatEscapesItsBoxForwardsNothing) {
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  cloud_pub_->publish(make_cloud({{2.0F, 0.0F, 0.0F}}, now));
  spin_for(400ms);
  EXPECT_EQ(received_.load(), 0) << "a hull the filter cannot trust must not configure the model";
  EXPECT_GT(filter_->totals().clouds, 0U);
}

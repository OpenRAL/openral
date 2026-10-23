// SPDX-License-Identifier: Apache-2.0
// The self-filter node, over real DDS: it forwards a cloud only when it knows
// where the robot was at the cloud's capture stamp, removes the robot's points
// when it does, and keeps the capture stamp on what it forwards.

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gtest/gtest.h>
#include <rcl_interfaces/srv/set_parameters_atomically.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/attached_collision_primitive.hpp>
#include <openral_msgs/msg/world_state_stamped.hpp>
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
    world_state_topic_ = "/test_sf_world_" + suffix;
    rclcpp::NodeOptions options;
    options.arguments({"--ros-args", "-r", "cloud_in:=" + in_, "-r", "cloud_out:=" + out_, "-r",
                       "cloud_out_rgb:=" + out_ + "_rgb"});
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
        {"world_state_topic", world_state_topic_},
        {"padding_m", padding_m()},
        {"max_joint_state_skew_s", 0.1},
    };
    for (auto& p : primitive_params()) {
      params.push_back(std::move(p));
    }
    for (auto& p : extra_params()) {
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
    world_state_pub_ = peer_->create_publisher<openral_msgs::msg::WorldStateStamped>(
        world_state_topic_, rclcpp::QoS(1).reliable());
    out_sub_ = peer_->create_subscription<sensor_msgs::msg::PointCloud2>(
        out_, rclcpp::QoS(2).reliable(), [this](sensor_msgs::msg::PointCloud2::SharedPtr m) {
          last_ = *m;
          ++received_;
        });
    executor_.add_node(filter_);
    executor_.add_node(peer_);
    spin_for(400ms);
  }

  virtual double padding_m() const { return 0.02; }
  virtual std::vector<rclcpp::Parameter> extra_params() const { return {}; }

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

  void publish_attachment(const std::string& evidence, const std::string& tracked_status = "bound",
                          const std::string& tracked_frame = "root", bool tracked_valid = true,
                          std::uint32_t tracked_epoch = 2, bool duplicate_track = false,
                          double state_age_s = 0.0) {
    openral_msgs::msg::WorldStateStamped state;
    const rclcpp::Time now = peer_->now();
    state.header.stamp = now - rclcpp::Duration::from_seconds(state_age_s);
    state.tracked_objects_valid = tracked_valid;
    state.tracked_objects.frame_id = tracked_frame;
    state.tracked_objects.schema_version = "1";
    openral_msgs::msg::AttachedCollisionObject object;
    object.object_id = "tracked:orange";
    object.attach_link = "l1";
    object.evidence_kind = evidence;
    object.epoch = 2;
    object.stamp_ns = now.nanoseconds();
    object.pose_in_link.position.x = 1.0;
    object.pose_in_link.orientation.w = 1.0;
    openral_msgs::msg::AttachedCollisionPrimitive primitive;
    primitive.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_BOX;
    primitive.shape_dimensions = {0.1, 0.1, 0.1};
    primitive.pose_in_object.orientation.w = 1.0;
    object.primitives.push_back(primitive);
    state.attached_objects.push_back(object);

    if (tracked_valid) {
      openral_msgs::msg::TrackedObject track;
      track.object_id = object.object_id;
      track.epoch = tracked_epoch;
      track.status = tracked_status;
      track.attach_link = object.attach_link;
      track.floor_z = -0.1;
      track.box.frame_id = tracked_frame;
      track.box.pose.position.x = 1.0;
      track.box.pose.orientation.w = 1.0;
      track.box.half_extents.x = 0.1;
      track.box.half_extents.y = 0.1;
      track.box.half_extents.z = 0.1;
      state.tracked_objects.objects.push_back(track);
      if (duplicate_track) {
        state.tracked_objects.objects.push_back(track);
      }
    }
    world_state_pub_->publish(state);
  }

  std::string in_, out_, js_, world_state_topic_;
  std::shared_ptr<bridge::RobotSelfFilter> filter_;
  std::shared_ptr<rclcpp::Node> peer_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_;
  rclcpp::Publisher<openral_msgs::msg::WorldStateStamped>::SharedPtr world_state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr out_sub_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  sensor_msgs::msg::PointCloud2 last_;
  std::atomic<int> received_{0};
};

}  // namespace

namespace {

class HALPeriodSelfFilterNode : public SelfFilterNode {
protected:
  std::vector<rclcpp::Parameter> extra_params() const override {
    return {{"joint_state_period_from_hal", true}, {"joint_state_period_s", 0.0}};
  }
};

}  // namespace

TEST_F(HALPeriodSelfFilterNode, DropsUntilActualPeriodAndFreshJointStateAreAdmitted) {
  const rclcpp::Time before_admission = peer_->now();
  send_joint_state(0.0, before_admission);
  spin_for(80ms);
  cloud_pub_->publish(make_cloud({{2.0F, 0.0F, 0.0F}}, before_admission));
  spin_for(160ms);
  EXPECT_EQ(received_.load(), 0) << "unadmitted cadence cannot produce map evidence";

  auto client = peer_->create_client<rcl_interfaces::srv::SetParametersAtomically>(
      "/openral_robot_self_filter/set_parameters_atomically");
  spin_for(250ms);
  ASSERT_TRUE(client->service_is_ready());
  auto request = std::make_shared<rcl_interfaces::srv::SetParametersAtomically::Request>();
  request->parameters = {
      rclcpp::Parameter("joint_state_period_s", std::numeric_limits<double>::quiet_NaN())
          .to_parameter_msg()};
  auto invalid_result = client->async_send_request(request);
  spin_for(200ms);
  ASSERT_EQ(invalid_result.wait_for(0s), std::future_status::ready);
  EXPECT_FALSE(invalid_result.get()->result.successful)
      << "a non-finite period must not open the fail-closed filter";

  request = std::make_shared<rcl_interfaces::srv::SetParametersAtomically::Request>();
  request->parameters = {rclcpp::Parameter("joint_state_period_s", 4.0 / 60.0).to_parameter_msg()};
  auto result = client->async_send_request(request);
  spin_for(200ms);
  ASSERT_EQ(result.wait_for(0s), std::future_status::ready);
  ASSERT_TRUE(result.get()->result.successful);

  const rclcpp::Time after_admission = peer_->now();
  cloud_pub_->publish(make_cloud({{2.0F, 0.0F, 0.0F}}, after_admission));
  spin_for(120ms);
  EXPECT_EQ(received_.load(), 0) << "pre-admission joint samples are discarded";
  send_joint_state(0.0, after_admission);
  spin_for(80ms);
  cloud_pub_->publish(make_cloud({{2.0F, 0.0F, 0.0F}}, after_admission));
  spin_for(250ms);
  EXPECT_EQ(received_.load(), 1) << "fresh state plus admitted actual cadence poses the cloud";
}

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

TEST_F(SelfFilterNode, GeometryOnlyAttachmentEvidenceKeepsSensorReturns) {
  const std::vector<std::string> evidence_kinds{"grasp_target_region", "vision_segmentation",
                                                "unknown_evidence", "pinch_fallback"};
  std::uint32_t expected_received = 0;
  for (const auto& evidence : evidence_kinds) {
    publish_attachment(evidence);
    spin_for(60ms);
    const rclcpp::Time stamp = peer_->now();
    send_joint_state(0.0, stamp);
    spin_for(60ms);
    cloud_pub_->publish(make_cloud({{1.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, stamp));
    spin_for(250ms);
    ++expected_received;
    ASSERT_EQ(received_.load(), expected_received) << evidence;
    EXPECT_EQ(last_.width, 2U) << evidence
                               << " geometry suppressed a sensor return without a tracked identity";
  }
}

TEST_F(SelfFilterNode, InvalidTrackedBindingKeepsSensorReturns) {
  const std::vector<std::function<void()>> invalid_states{
      [this]() { publish_attachment("tracked_object", "bound", "root", false); },
      [this]() { publish_attachment("tracked_object", "bound", "map"); },
      [this]() { publish_attachment("tracked_object", "bound", "root", true, 1); },
      [this]() { publish_attachment("tracked_object", "bound", "root", true, 2, true); },
      [this]() { publish_attachment("tracked_object", "bound", "root", true, 2, false, 0.6); },
  };
  std::uint32_t expected_received = 0;
  for (const auto& publish : invalid_states) {
    publish();
    spin_for(60ms);
    const rclcpp::Time stamp = peer_->now();
    send_joint_state(0.0, stamp);
    spin_for(60ms);
    cloud_pub_->publish(make_cloud({{1.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, stamp));
    spin_for(250ms);
    ++expected_received;
    ASSERT_EQ(received_.load(), expected_received);
    EXPECT_EQ(last_.width, 2U) << "an unusable or ambiguous tracked binding filtered points";
  }
}

TEST_F(SelfFilterNode, UniqueBoundTrackedObjectFiltersItsOwnReturns) {
  publish_attachment("tracked_object");
  spin_for(60ms);
  const rclcpp::Time stamp = peer_->now();
  send_joint_state(0.0, stamp);
  spin_for(60ms);
  cloud_pub_->publish(make_cloud({{1.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, stamp));
  spin_for(250ms);

  ASSERT_EQ(received_.load(), 1U);
  ASSERT_EQ(last_.width, 1U);
  float x = 0.0F;
  std::memcpy(&x, last_.data.data(), sizeof(float));
  EXPECT_FLOAT_EQ(x, 2.0F);
}

TEST_F(SelfFilterNode, AColouredCloudReachesTheMapAsXyzAndViewersWithItsColour) {
  // Review G: the map never carries colour; a viewer gets it, robot removed.
  sensor_msgs::msg::PointCloud2 rgb_last;
  std::atomic<int> rgb_received{0};
  auto rgb_sub = peer_->create_subscription<sensor_msgs::msg::PointCloud2>(
      out_ + "_rgb", rclcpp::QoS(2).reliable(), [&](sensor_msgs::msg::PointCloud2::SharedPtr m) {
        rgb_last = *m;
        ++rgb_received;
      });
  spin_for(300ms);
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  auto cloud = make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, now);
  // Re-pack as x, y, z, rgb (16 bytes), as the sim bridge publishes a coloured cloud.
  std::vector<std::uint8_t> packed(2 * 16);
  for (std::size_t i = 0; i < 2; ++i) {
    std::memcpy(packed.data() + i * 16, cloud.data.data() + i * 12, 12);
    const std::uint32_t colour = 0x00FF8000U + static_cast<std::uint32_t>(i);
    std::memcpy(packed.data() + i * 16 + 12, &colour, 4);
  }
  sensor_msgs::msg::PointField rgb;
  rgb.name = "rgb";
  rgb.offset = 12;
  rgb.datatype = sensor_msgs::msg::PointField::FLOAT32;
  rgb.count = 1;
  cloud.fields.push_back(rgb);
  cloud.point_step = 16;
  cloud.row_step = 32;
  cloud.data = packed;
  cloud_pub_->publish(cloud);
  spin_for(400ms);

  ASSERT_EQ(received_.load(), 1);
  EXPECT_EQ(last_.point_step, 12U) << "the map's cloud carries x/y/z only";
  ASSERT_EQ(last_.fields.size(), 3U);
  ASSERT_EQ(last_.width, 1U);
  float x = 0.0F;
  std::memcpy(&x, last_.data.data(), sizeof(float));
  EXPECT_FLOAT_EQ(x, 2.0F);

  ASSERT_EQ(rgb_received.load(), 1) << "a viewer gets the coloured, robot-free cloud";
  EXPECT_EQ(rgb_last.point_step, 16U);
  ASSERT_EQ(rgb_last.width, 1U);
  std::uint32_t colour = 0;
  std::memcpy(&colour, rgb_last.data.data() + 12, 4);
  EXPECT_EQ(colour, 0x00FF8001U) << "the world point keeps its own colour";
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

TEST_F(SelfFilterNode, InterpolatesTheJointAtTheCloudStamp) {
  // q = 0 at t0 and pi/2 at t0 + 40 ms (default period 50 ms): at t0 + 20 ms the
  // capsule sits at 45 degrees, where neither sample puts it.
  const rclcpp::Time t0 = peer_->now();
  send_joint_state(0.0, t0);
  send_joint_state(M_PI / 2.0, t0 + rclcpp::Duration(40ms));
  spin_for(100ms);
  const float c = static_cast<float>(0.5 / std::sqrt(2.0));
  cloud_pub_->publish(make_cloud({{c, c, 0.0F}, {0.5F, 0.0F, 0.0F}, {0.0F, 0.5F, 0.0F}},
                                 t0 + rclcpp::Duration(20ms)));
  spin_for(400ms);
  ASSERT_EQ(received_.load(), 1);
  EXPECT_EQ(last_.width, 2U) << "only the interpolated pose is robot";
}

namespace {

// Issue #418 (Spark R16): the robot's recent sweep pads the filter too. The capsule
// rests at q = 0, turns to 0.15 rad by t0 + 160 ms and stops there. The cloud at
// t0 + 200 ms holds a return where the capsule sat (its old centre, 25 mm outside
// the capsule's 20 mm padding now) and a world point.
class TurnNode : public SelfFilterNode {
protected:
  void publish_turn_and_cloud() {
    const rclcpp::Time t0 = peer_->now();
    const double q[] = {0.0, 0.0, 0.05, 0.1, 0.15, 0.15, 0.15};
    for (int i = 0; i < 7; ++i) {
      send_joint_state(q[i], t0 + rclcpp::Duration(std::chrono::milliseconds(40 * i)));
    }
    spin_for(100ms);
    cloud_pub_->publish(
        make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, t0 + rclcpp::Duration(200ms)));
    spin_for(400ms);
  }
};

class TrailingSweepNode : public TurnNode {
protected:
  std::vector<rclcpp::Parameter> extra_params() const override {
    return {{"trailing_sweep_s", 0.2}};
  }
};

class OverCapTrailingSweepNode : public TurnNode {
protected:
  std::vector<rclcpp::Parameter> extra_params() const override {
    return {{"trailing_sweep_s", bridge::kMaxTrailingSweepS + 1e-6}};
  }
};

}  // namespace

TEST_F(TurnNode, WithoutTheTrailingSweepTheReturnTheRobotJustLeftIsKept) {
  publish_turn_and_cloud();
  ASSERT_EQ(received_.load(), 1);
  EXPECT_EQ(last_.width, 2U) << "today: only the pose at the stamp (still) is padded";
}

TEST_F(TrailingSweepNode, ARobotReturnInTheRecentSweepIsDroppedAndTheWorldKept) {
  publish_turn_and_cloud();
  ASSERT_EQ(received_.load(), 1);
  EXPECT_EQ(last_.width, 1U) << "the return where the capsule was 0.2 s ago is the robot's";
}

TEST_F(TrailingSweepNode, WithNoStateThatFarBackThePaddingIsTodays) {
  // The history starts at the turn: nothing reaches 0.2 s back, so nothing extra is
  // dropped (a fresh start keeps robot points, it never erases the world).
  const rclcpp::Time t0 = peer_->now();
  const double q[] = {0.1, 0.15, 0.15};
  for (int i = 0; i < 3; ++i) {
    send_joint_state(q[i], t0 + rclcpp::Duration(std::chrono::milliseconds(40 * i)));
  }
  spin_for(100ms);
  cloud_pub_->publish(
      make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, t0 + rclcpp::Duration(80ms)));
  spin_for(400ms);
  ASSERT_EQ(received_.load(), 1);
  EXPECT_EQ(last_.width, 2U);
}

TEST_F(OverCapTrailingSweepNode, ATrailingSweepPastTheCapForwardsNothing) {
  publish_turn_and_cloud();
  EXPECT_EQ(received_.load(), 0);
}

TEST_F(SelfFilterNode, RefusesACloudBetweenJointStatesTooFarApart) {
  const rclcpp::Time t0 = peer_->now();
  send_joint_state(0.0, t0);
  send_joint_state(0.0, t0 + rclcpp::Duration(200ms));  // 4 periods apart
  spin_for(100ms);
  cloud_pub_->publish(make_cloud({{2.0F, 0.0F, 0.0F}}, t0 + rclcpp::Duration(100ms)));
  spin_for(400ms);
  EXPECT_EQ(received_.load(), 0) << "a refused frame inserts nothing";
  EXPECT_GT(filter_->totals().refused_joint_gap, 0U);
}

namespace {

// `padding_m` is capped in the node itself (DeployRuntime's cap, hazard log
// Entry 035), so a filter launched outside `openral deploy` cannot widen the
// blind shell around the arm.
class AtCapPaddingNode : public SelfFilterNode {
protected:
  double padding_m() const override { return bridge::kMaxSelfFilterPaddingM; }
};
class OverCapPaddingNode : public SelfFilterNode {
protected:
  double padding_m() const override { return bridge::kMaxSelfFilterPaddingM + 1e-6; }
};
class NegativePaddingNode : public SelfFilterNode {
protected:
  double padding_m() const override { return -0.01; }
};
class NanPaddingNode : public SelfFilterNode {
protected:
  double padding_m() const override { return std::numeric_limits<double>::quiet_NaN(); }
};

}  // namespace

TEST_F(AtCapPaddingNode, APaddingAtTheCapIsAccepted) {
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  // Capsule surface at x = 0.55: 0.64 is inside a 0.10 m shell, 2.0 is world.
  cloud_pub_->publish(make_cloud({{0.64F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, now));
  spin_for(400ms);
  ASSERT_EQ(received_.load(), 1);
  EXPECT_EQ(last_.width, 1U);
}

TEST_F(OverCapPaddingNode, APaddingPastTheCapForwardsNothing) {
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  cloud_pub_->publish(make_cloud({{2.0F, 0.0F, 0.0F}}, now));
  spin_for(400ms);
  EXPECT_EQ(received_.load(), 0) << "a refused padding must not configure the filter";
  EXPECT_GT(filter_->totals().clouds, 0U);
}

TEST_F(NegativePaddingNode, ANegativePaddingForwardsNothing) {
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  cloud_pub_->publish(make_cloud({{2.0F, 0.0F, 0.0F}}, now));
  spin_for(400ms);
  EXPECT_EQ(received_.load(), 0) << "a refused padding must not configure the filter";
}

TEST_F(NanPaddingNode, ANonFinitePaddingForwardsNothing) {
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  cloud_pub_->publish(make_cloud({{2.0F, 0.0F, 0.0F}}, now));
  spin_for(400ms);
  EXPECT_EQ(received_.load(), 0) << "a refused padding must not configure the filter";
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

namespace {

// One depth camera, `depth_cam`, 0.5 m behind the capsule along its +z: the
// capsule (r = 0.05, segment 0.45..0.55 m ahead) fills the image centre.
class RobotMaskNode : public SelfFilterNode {
protected:
  std::vector<rclcpp::Parameter> extra_params() const override {
    const std::string s = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    return {
        {"robot_mask_enabled", true},
        {"robot_mask_cameras", std::vector<std::string>{"depth_cam"}},
        {"robot_mask_depth_topics", std::vector<std::string>{"/test_rm_depth_" + s}},
        {"robot_mask_camera_info_topics", std::vector<std::string>{"/test_rm_info_" + s}},
        {"robot_mask_topics", std::vector<std::string>{"/test_rm_mask_" + s}},
        {"robot_mask_uncertain_px", std::int64_t{2}},
    };
  }
};

}  // namespace

TEST_F(RobotMaskNode, PublishesTheRobotSilhouetteWithTheDepthStamp) {
  const std::string s = ::testing::UnitTest::GetInstance()->current_test_info()->name();
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = peer_->now();
  tf.header.frame_id = "depth_cam";
  tf.child_frame_id = "root";
  tf.transform.translation.x = -0.5;
  tf.transform.translation.z = 0.5;
  tf.transform.rotation.w = 1.0;
  tf_->sendTransform(tf);
  auto info_pub = peer_->create_publisher<sensor_msgs::msg::CameraInfo>("/test_rm_info_" + s,
                                                                        rclcpp::QoS(1).reliable());
  auto depth_pub = peer_->create_publisher<sensor_msgs::msg::Image>(
      "/test_rm_depth_" + s, rclcpp::SensorDataQoS().keep_last(2));
  sensor_msgs::msg::Image mask;
  std::atomic<int> masks{0};
  auto mask_sub = peer_->create_subscription<sensor_msgs::msg::Image>(
      "/test_rm_mask_" + s, rclcpp::SensorDataQoS(), [&](sensor_msgs::msg::Image::SharedPtr m) {
        mask = *m;
        ++masks;
      });
  spin_for(300ms);

  sensor_msgs::msg::CameraInfo info;
  info.width = 64;
  info.height = 48;
  info.k = {100.0, 0.0, 32.0, 0.0, 100.0, 24.0, 0.0, 0.0, 1.0};
  info_pub->publish(info);
  const rclcpp::Time now = peer_->now();
  send_joint_state(0.0, now);
  spin_for(100ms);
  sensor_msgs::msg::Image depth;
  depth.header.stamp = now;
  depth.header.frame_id = "depth_cam";
  depth.width = 64;
  depth.height = 48;
  depth.encoding = "32FC1";
  depth.step = 64 * 4;
  // A wall behind the arm, and the arm (the capsule, front at 0.45 m) where it is seen.
  std::vector<float> metres(64 * 48, 1.0F);
  for (int v = 20; v <= 28; ++v) {
    for (int u = 28; u <= 36; ++u) {
      metres[static_cast<std::size_t>(v * 64 + u)] = 0.47F;
    }
  }
  depth.data.resize(depth.step * depth.height);
  std::memcpy(depth.data.data(), metres.data(), depth.data.size());
  depth_pub->publish(depth);
  spin_for(400ms);

  ASSERT_EQ(masks.load(), 1) << "dropped: " << filter_->totals().masks_dropped;
  EXPECT_EQ(rclcpp::Time(mask.header.stamp).nanoseconds(), now.nanoseconds())
      << "stamped with the DEPTH stamp";
  EXPECT_EQ(mask.encoding, "mono8");
  ASSERT_EQ(mask.data.size(), 64U * 48U);
  EXPECT_EQ(mask.data[24 * 64 + 32], bridge::kRobotMaskRobot);
  EXPECT_EQ(mask.data[0], bridge::kRobotMaskFree);

  // A depth image at a stamp the joint states cannot reach publishes nothing.
  depth.header.stamp = now + rclcpp::Duration(1s);
  depth_pub->publish(depth);
  spin_for(300ms);
  EXPECT_EQ(masks.load(), 1) << "a refused frame is no evidence";
  EXPECT_GT(filter_->totals().masks_dropped, 0U);
}

namespace {
class DelayedMaskNode : public RobotMaskNode {
protected:
  void SetUp() override {
    RobotMaskNode::SetUp();
    const std::string s = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    info_pub_ = peer_->create_publisher<sensor_msgs::msg::CameraInfo>("/test_rm_info_" + s,
                                                                      rclcpp::QoS(1).reliable());
    depth_pub_ = peer_->create_publisher<sensor_msgs::msg::Image>("/test_rm_depth_" + s,
                                                                  rclcpp::SensorDataQoS());
    mask_sub_ = peer_->create_subscription<sensor_msgs::msg::Image>(
        "/test_rm_mask_" + s, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Image::SharedPtr msg) {
          mask_ = *msg;
          ++masks_;
        });
    dynamic_tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(peer_);
    spin_for(300ms);
    sensor_msgs::msg::CameraInfo info;
    info.width = 64;
    info.height = 48;
    info.k = {100.0, 0.0, 32.0, 0.0, 100.0, 24.0, 0.0, 0.0, 1.0};
    info_pub_->publish(info);
    spin_for(30ms);
  }

  rclcpp::Time send_depth() {
    const auto stamp = peer_->now();
    send_joint_state(0.0, stamp);
    spin_for(10ms);
    sensor_msgs::msg::Image depth;
    depth.header.stamp = stamp;
    depth.header.frame_id = "delayed_cam";
    depth.width = 64;
    depth.height = 48;
    depth.encoding = "32FC1";
    depth.step = 64 * 4;
    std::vector<float> metres(64 * 48, 1.0F);
    for (int v = 20; v <= 28; ++v) {
      for (int u = 28; u <= 36; ++u) {
        metres[static_cast<std::size_t>(v * 64 + u)] = 0.47F;
      }
    }
    depth.data.resize(depth.step * depth.height);
    std::memcpy(depth.data.data(), metres.data(), depth.data.size());
    depth_pub_->publish(depth);
    spin_for(15ms);
    return stamp;
  }

  void send_camera_tf(const rclcpp::Time& stamp) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = "delayed_cam";
    tf.child_frame_id = "cam";
    tf.transform.translation.x = -0.5;
    tf.transform.translation.z = 0.5;
    tf.transform.rotation.w = 1.0;
    dynamic_tf_->sendTransform(tf);
  }

  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> dynamic_tf_;
  sensor_msgs::msg::Image mask_;
  int masks_{0};
};
}  // namespace

TEST_F(DelayedMaskNode, ExactStampTfArrivingAfterDepthProducesTheSourceStampedMask) {
  const auto stamp = send_depth();
  ASSERT_EQ(masks_, 0);
  ASSERT_EQ(filter_->totals().masks_dropped, 0U);
  send_camera_tf(stamp - rclcpp::Duration(50ms));
  spin_for(10ms);
  EXPECT_EQ(masks_, 0) << "latest available TF is not evidence at the depth stamp";
  send_camera_tf(stamp);
  spin_for(70ms);
  ASSERT_EQ(masks_, 1);
  EXPECT_EQ(rclcpp::Time(mask_.header.stamp).nanoseconds(), stamp.nanoseconds());
  EXPECT_EQ(mask_.header.frame_id, "delayed_cam");
  ASSERT_EQ(mask_.data.size(), 64U * 48U);
  EXPECT_EQ(mask_.data[24 * 64 + 32], bridge::kRobotMaskRobot);
  EXPECT_EQ(mask_.data[0], bridge::kRobotMaskFree);
}

TEST_F(DelayedMaskNode, MissingTfExpiresAndLateTfCannotResurrectTheDepth) {
  const auto stamp = send_depth();
  spin_for(150ms);
  EXPECT_EQ(masks_, 0);
  EXPECT_EQ(filter_->totals().masks_dropped, 1U);
  send_camera_tf(stamp);
  spin_for(100ms);
  EXPECT_EQ(masks_, 0);
  EXPECT_EQ(filter_->totals().masks_dropped, 1U);
}

TEST_F(DelayedMaskNode, PendingDepthQueueRefusesOverflowAndExpiresTheRemainder) {
  send_depth();
  send_depth();
  send_depth();
  EXPECT_EQ(filter_->totals().masks_dropped, 1U);
  spin_for(150ms);
  EXPECT_EQ(filter_->totals().masks_dropped, 3U);
  EXPECT_EQ(masks_, 0);
}

namespace {
class ZeroWaitMaskNode : public DelayedMaskNode {
protected:
  std::vector<rclcpp::Parameter> extra_params() const override {
    auto params = RobotMaskNode::extra_params();
    params.emplace_back("max_joint_state_skew_s", 0.0);
    return params;
  }
};
}  // namespace

TEST_F(ZeroWaitMaskNode, ZeroSkewStillPublishesWithAnExactPoseAndTf) {
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = peer_->now();
  tf.header.frame_id = "delayed_cam";
  tf.child_frame_id = "cam";
  tf.transform.translation.x = -0.5;
  tf.transform.translation.z = 0.5;
  tf.transform.rotation.w = 1.0;
  tf_->sendTransform(tf);
  spin_for(30ms);
  const auto stamp = send_depth();
  spin_for(20ms);
  ASSERT_EQ(masks_, 1);
  EXPECT_EQ(rclcpp::Time(mask_.header.stamp).nanoseconds(), stamp.nanoseconds());
  EXPECT_EQ(filter_->totals().masks_dropped, 0U);
}

TEST_F(SelfFilterNode, CloudWaitsForCapturedJointStateInsteadOfExtrapolating) {
  const auto t0 = peer_->now();
  send_joint_state(-M_PI / 4.0, t0 - rclcpp::Duration(20ms));
  send_joint_state(0.0, t0);
  spin_for(20ms);
  const auto captured = t0 + rclcpp::Duration(20ms);
  cloud_pub_->publish(make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, captured));
  spin_for(20ms);
  ASSERT_EQ(received_.load(), 0) << "two earlier samples do not prove the captured robot pose";

  send_joint_state(0.0, captured);
  spin_for(60ms);
  ASSERT_EQ(received_.load(), 1);
  ASSERT_EQ(last_.width, 1U);
  float x = 0.0F;
  std::memcpy(&x, last_.data.data(), sizeof(float));
  EXPECT_FLOAT_EQ(x, 2.0F) << "the captured pose removes robot returns and preserves the world";
  EXPECT_EQ(rclcpp::Time(last_.header.stamp).nanoseconds(), captured.nanoseconds());
}

TEST_F(SelfFilterNode, PendingCloudUsesALaterBracketingJointState) {
  const auto t0 = peer_->now();
  send_joint_state(0.0, t0);
  spin_for(20ms);
  const auto captured = t0 + rclcpp::Duration(20ms);
  const float c = static_cast<float>(0.5 / std::sqrt(2.0));
  cloud_pub_->publish(make_cloud({{c, c, 0.0F}, {2.0F, 0.0F, 0.0F}}, captured));
  spin_for(20ms);
  ASSERT_EQ(received_.load(), 0);
  send_joint_state(M_PI / 2.0, t0 + rclcpp::Duration(40ms));
  spin_for(60ms);
  ASSERT_EQ(received_.load(), 1);
  ASSERT_EQ(last_.width, 1U) << "the delayed bracket must interpolate, not use either endpoint";
  float x = 0.0F;
  std::memcpy(&x, last_.data.data(), sizeof(float));
  EXPECT_FLOAT_EQ(x, 2.0F);
  EXPECT_EQ(rclcpp::Time(last_.header.stamp).nanoseconds(), captured.nanoseconds());
}

TEST_F(SelfFilterNode, ExpiredCloudCannotBeReleasedByALateJointState) {
  const auto t0 = peer_->now();
  send_joint_state(0.0, t0);
  spin_for(20ms);
  const auto captured = t0 + rclcpp::Duration(20ms);
  cloud_pub_->publish(make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, captured));
  spin_for(160ms);
  ASSERT_EQ(received_.load(), 0);
  send_joint_state(0.0, captured);
  spin_for(100ms);
  EXPECT_EQ(received_.load(), 0) << "late state must not resurrect expired sensor evidence";
}

TEST_F(SelfFilterNode, PendingCloudOverflowDropsOldestAndReleasesOnlyTwo) {
  const auto t0 = peer_->now();
  send_joint_state(0.0, t0);
  spin_for(20ms);
  for (const auto offset : {20ms, 30ms, 40ms}) {
    cloud_pub_->publish(
        make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, t0 + rclcpp::Duration(offset)));
    spin_for(10ms);
  }
  ASSERT_EQ(received_.load(), 0);
  send_joint_state(0.0, t0 + rclcpp::Duration(40ms));
  spin_for(60ms);
  EXPECT_EQ(received_.load(), 2) << "the bounded queue must not release the evicted cloud";
  EXPECT_EQ(last_.width, 1U);
  EXPECT_EQ(rclcpp::Time(last_.header.stamp).nanoseconds(),
            (t0 + rclcpp::Duration(40ms)).nanoseconds());
}

TEST_F(SelfFilterNode, CloudOlderThanJointHistoryIsNotBackwardExtrapolated) {
  const auto captured = peer_->now();
  send_joint_state(0.0, captured + rclcpp::Duration(20ms));
  spin_for(20ms);
  cloud_pub_->publish(make_cloud({{0.5F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}, captured));
  spin_for(160ms);
  EXPECT_EQ(received_.load(), 0) << "a later pose cannot prove the robot's earlier geometry";
  EXPECT_GT(filter_->totals().dropped_no_state, 0U);
}

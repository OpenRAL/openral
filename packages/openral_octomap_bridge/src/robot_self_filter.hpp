// SPDX-License-Identifier: Apache-2.0
// ROS node: removes the robot's own returns (and a held payload's) from a real
// depth cloud before `octomap_server` inserts it. The geometry and the reasons
// are in `openral_octomap_bridge/self_filter.hpp`; this file is the plumbing.
//
// Poses are taken AT THE CLOUD'S CAPTURE STAMP, never "latest": the arm's joint
// positions from a short history of joint-state messages, and the camera's pose
// in the model root from tf2 at that stamp. A moving arm filtered with a pose
// from a different instant leaves a ghost trail of its own points in the map.
//
// Every way of not knowing where the robot was when the cloud was captured —
// no joint state within `max_joint_state_skew_s` of the stamp, a joint the
// model needs never reported, no camera transform at the stamp — DROPS THE
// CLOUD rather than forwarding it unfiltered or filtered against a guessed
// pose. A dropped cloud is silence downstream: the octree goes stale, the
// voxel bridge stops republishing (`max_octree_age_s`), and the kernel's world
// check fails closed. Forwarding it unfiltered would fill the map with the arm
// and stop the robot against itself; filtering it against a stale pose would
// punch the blind shell into the wrong place, which is the unsafe direction.
//
// A held payload (from `/openral/world_state_fast`'s `attached_objects`) is
// removed the same way. Anything about it that cannot be placed — a stale
// snapshot, an attach link neither in the model nor in TF, geometry
// `place_attached_object` refuses — leaves its points IN the cloud: the map
// keeps occupancy it may not need, which can only stop the robot early.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <openral_msgs/msg/world_state_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "openral_octomap_bridge/payload_clearing.hpp"
#include "openral_octomap_bridge/self_filter.hpp"

namespace openral_octomap_bridge {

class RobotSelfFilter : public rclcpp::Node {
public:
  explicit RobotSelfFilter(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : rclcpp::Node("openral_robot_self_filter", options) {
    SelfModelParams params;
    params.n_links = this->declare_parameter<std::int64_t>("collision_n_links", 0);
    params.parent = declare_ints("collision_parent");
    params.joint_kind = declare_ints("collision_joint_kind");
    params.dof_index = declare_ints("collision_dof_index");
    params.origin_xyzrpy = declare_doubles("collision_origin_xyzrpy");
    params.axis = declare_doubles("collision_axis");
    params.link_names = declare_strings("collision_link_names");
    params.capsule_link = declare_ints("collision_capsule_link");
    params.capsule_radius = declare_doubles("collision_capsule_radius");
    params.capsule_half_length = declare_doubles("collision_capsule_half_length");
    params.capsule_origin_xyzrpy = declare_doubles("collision_capsule_origin_xyzrpy");
    params.box_link = declare_ints("collision_box_link");
    params.box_half_extents = declare_doubles("collision_box_half_extents");
    params.box_origin_xyzrpy = declare_doubles("collision_box_origin_xyzrpy");
    const auto joint_names = declare_strings("collision_joint_names");
    // Parallel to `collision_joint_names`: a second spelling each dof answers
    // to (the manifest's `sim_joint_name`, which is the upstream name a vendor
    // `/joint_states` uses). Empty entries are ignored.
    const auto joint_aliases = declare_strings("collision_joint_aliases");

    padding_m_ = this->declare_parameter<double>("padding_m", 0.05);
    max_skew_s_ = this->declare_parameter<double>("max_joint_state_skew_s", 0.1);
    tf_timeout_s_ = this->declare_parameter<double>("tf_timeout_s", 0.05);
    history_s_ = this->declare_parameter<double>("joint_state_history_s", 2.0);
    attached_enabled_ = this->declare_parameter<bool>("attached_filter_enabled", true);
    attached_timeout_s_ = this->declare_parameter<double>("attached_state_timeout_s", 0.5);
    const auto joint_states_topic =
        this->declare_parameter<std::string>("joint_states_topic", "/joint_states");
    const auto world_state_topic =
        this->declare_parameter<std::string>("world_state_topic", "/openral/world_state_fast");

    std::string error;
    if (!build_self_model(params, model_, error)) {
      // Fail closed: with no model the node forwards nothing, so octomap gets
      // no input and the kernel's world check drops every chunk. Better than
      // an unfiltered map that stops the robot against itself on tick 1.
      RCLCPP_ERROR(this->get_logger(),
                   "robot self-filter has no usable collision model (%s): forwarding NOTHING",
                   error.c_str());
    } else {
      model_ok_ = true;
    }
    if (joint_names.size() < model_.required_dof) {
      RCLCPP_ERROR(this->get_logger(),
                   "collision_joint_names has %zu names but the model uses %zu dofs: "
                   "forwarding NOTHING",
                   joint_names.size(), model_.required_dof);
      model_ok_ = false;
    }
    for (std::size_t d = 0; d < joint_names.size(); ++d) {
      dof_of_name_.emplace(joint_names[d], d);
    }
    for (std::size_t d = 0; d < joint_aliases.size() && d < joint_names.size(); ++d) {
      if (!joint_aliases[d].empty()) {
        dof_of_name_.emplace(joint_aliases[d], d);  // never overrides a manifest name
      }
    }
    for (const int dof : model_.dof_index) {
      if (dof >= 0) {
        required_dofs_.push_back(static_cast<std::size_t>(dof));
      }
    }
    q_.assign(std::max(model_.required_dof, joint_names.size()), 0.0);
    seen_.assign(q_.size(), 0);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        joint_states_topic, rclcpp::SensorDataQoS().keep_last(50),
        [this](sensor_msgs::msg::JointState::SharedPtr msg) { on_joint_state(msg); });
    world_state_sub_ = this->create_subscription<openral_msgs::msg::WorldStateStamped>(
        world_state_topic, rclcpp::QoS(1).reliable(),
        [this](openral_msgs::msg::WorldStateStamped::SharedPtr msg) {
          world_state_ = std::move(msg);
        });
    // Reliable out: compatible with a reliable or a best-effort octomap input.
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_out",
                                                                       rclcpp::QoS(2).reliable());
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "cloud_in", rclcpp::SensorDataQoS().keep_last(2),
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { on_cloud(msg); });
    stats_timer_ = this->create_wall_timer(std::chrono::seconds(5), [this]() { log_stats(); });

    RCLCPP_INFO(this->get_logger(),
                "robot self-filter: %zu links, %zu primitives, padding %.3f m, root %s, "
                "joint states %s (skew <= %.3f s)",
                model_.n_links(), model_.primitives.size(), padding_m_,
                model_.link_names.empty() ? "?" : model_.link_names.front().c_str(),
                joint_states_topic.c_str(), max_skew_s_);
  }

  /// Totals since start, for tests and the stats log.
  struct Totals {
    std::uint64_t clouds{0};
    std::uint64_t published{0};
    std::uint64_t dropped_no_state{0};
    std::uint64_t dropped_no_tf{0};
    std::uint64_t dropped_bad_cloud{0};
    std::uint64_t points_in{0};
    std::uint64_t removed{0};
    std::uint64_t payload_objects_filtered{0};
  };
  const Totals& totals() const noexcept { return totals_; }

private:
  struct Snapshot {
    std::int64_t stamp_ns{0};
    std::vector<double> q;
    bool complete{false};
  };

  std::vector<std::int64_t> declare_ints(const std::string& name) {
    return this->declare_parameter<std::vector<std::int64_t>>(name, std::vector<std::int64_t>{});
  }
  std::vector<double> declare_doubles(const std::string& name) {
    return this->declare_parameter<std::vector<double>>(name, std::vector<double>{});
  }
  std::vector<std::string> declare_strings(const std::string& name) {
    return this->declare_parameter<std::vector<std::string>>(name, std::vector<std::string>{});
  }

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr& msg) {
    const std::size_t n = std::min(msg->name.size(), msg->position.size());
    for (std::size_t i = 0; i < n; ++i) {
      const auto it = dof_of_name_.find(msg->name[i]);
      if (it == dof_of_name_.end() || it->second >= q_.size()) {
        continue;
      }
      q_[it->second] = msg->position[i];
      seen_[it->second] = 1;
    }
    Snapshot snap;
    snap.stamp_ns =
        rclcpp::Time(msg->header.stamp, this->get_clock()->get_clock_type()).nanoseconds();
    snap.q = q_;
    snap.complete = std::all_of(required_dofs_.begin(), required_dofs_.end(),
                                [this](std::size_t d) { return seen_[d] != 0; });
    history_.push_back(std::move(snap));
    const auto horizon_ns = static_cast<std::int64_t>(history_s_ * 1e9);
    while (!history_.empty() && history_.back().stamp_ns - history_.front().stamp_ns > horizon_ns) {
      history_.pop_front();
    }
  }

  /// The complete snapshot nearest `stamp_ns`, if it is within the skew bound.
  const Snapshot* snapshot_at(std::int64_t stamp_ns) const {
    const Snapshot* best = nullptr;
    std::int64_t best_dt = 0;
    for (const auto& s : history_) {
      if (!s.complete) {
        continue;
      }
      const std::int64_t dt = std::llabs(s.stamp_ns - stamp_ns);
      if (best == nullptr || dt < best_dt) {
        best = &s;
        best_dt = dt;
      }
    }
    if (best == nullptr || static_cast<double>(best_dt) * 1e-9 > max_skew_s_) {
      return nullptr;
    }
    return best;
  }

  static bool float_field(const sensor_msgs::msg::PointCloud2& cloud, const char* name,
                          std::size_t& offset) {
    for (const auto& f : cloud.fields) {
      if (f.name == name) {
        if (f.datatype != sensor_msgs::msg::PointField::FLOAT32 || f.count != 1) {
          return false;
        }
        offset = f.offset;
        return true;
      }
    }
    return false;
  }

  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    ++totals_.clouds;
    if (!model_ok_) {
      return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const rclcpp::Time stamp(msg->header.stamp, this->get_clock()->get_clock_type());

    const Snapshot* snap = snapshot_at(stamp.nanoseconds());
    if (snap == nullptr) {
      ++totals_.dropped_no_state;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "no complete joint state within %.3f s of the cloud stamp: dropping "
                           "the cloud (history %zu)",
                           max_skew_s_, history_.size());
      return;
    }
    tf2::Transform cloud_from_root;
    const std::string& root = model_.link_names.front();
    try {
      const auto tf_msg = tf_buffer_->lookupTransform(
          msg->header.frame_id, root, stamp, rclcpp::Duration::from_seconds(tf_timeout_s_));
      tf2::fromMsg(tf_msg.transform, cloud_from_root);
    } catch (const tf2::TransformException& ex) {
      ++totals_.dropped_no_tf;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "TF %s <- %s at the cloud stamp unavailable (%s): dropping the cloud",
                           msg->header.frame_id.c_str(), root.c_str(), ex.what());
      return;
    }

    self_forward_kinematics(model_, snap->q.data(), snap->q.size(), link_world_);
    primitives_.clear();
    place_self_primitives(model_, link_world_, cloud_from_root, primitives_);
    add_payload_primitives(stamp, cloud_from_root);
    mask_.set(primitives_, padding_m_);

    std::size_t xo = 0;
    std::size_t yo = 0;
    std::size_t zo = 0;
    const std::size_t n_points = static_cast<std::size_t>(msg->width) * msg->height;
    if (!float_field(*msg, "x", xo) || !float_field(*msg, "y", yo) || !float_field(*msg, "z", zo) ||
        msg->point_step < 12 || msg->data.size() < n_points * msg->point_step) {
      ++totals_.dropped_bad_cloud;
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                            "cloud has no FLOAT32 x/y/z or a short data buffer: dropping it");
      return;
    }
    auto out = std::make_unique<sensor_msgs::msg::PointCloud2>();
    out->header = msg->header;  // the CAPTURE stamp travels on, into the octree
    out->fields = msg->fields;
    out->is_bigendian = msg->is_bigendian;
    out->point_step = msg->point_step;
    const auto stats = filter_xyz_points(msg->data.data(), n_points, msg->point_step, xo, yo, zo,
                                         mask_, out->data);
    out->height = 1;
    out->width = static_cast<std::uint32_t>(stats.kept);
    out->row_step = out->point_step * out->width;
    out->is_dense = true;
    cloud_pub_->publish(std::move(out));

    ++totals_.published;
    totals_.points_in += stats.points_in - stats.non_finite;
    totals_.removed += stats.removed;
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    window_ms_sum_ += ms;
    window_ms_max_ = std::max(window_ms_max_, ms);
    ++window_clouds_;
  }

  void add_payload_primitives(const rclcpp::Time& stamp, const tf2::Transform& cloud_from_root) {
    const auto state = world_state_;
    if (!attached_enabled_ || state == nullptr || state->attached_objects.empty()) {
      return;
    }
    const rclcpp::Time state_stamp(state->header.stamp, this->get_clock()->get_clock_type());
    if ((this->now() - state_stamp).seconds() > attached_timeout_s_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "attachment state is stale: the held payload stays in the cloud");
      return;
    }
    std::vector<SupportPatch> patches;  // the grid bridge's concern, not the cloud's
    for (const auto& object : state->attached_objects) {
      tf2::Transform root_from_link;
      const int link = model_.link_index(object.attach_link);
      if (link >= 0) {
        root_from_link = link_world_[static_cast<std::size_t>(link)];
      } else {
        try {
          const auto tf_msg =
              tf_buffer_->lookupTransform(model_.link_names.front(), object.attach_link, stamp,
                                          rclcpp::Duration::from_seconds(tf_timeout_s_));
          tf2::fromMsg(tf_msg.transform, root_from_link);
        } catch (const tf2::TransformException&) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "attach link %s is neither a model link nor in TF: payload %s "
                               "stays in the cloud",
                               object.attach_link.c_str(), object.object_id.c_str());
          continue;
        }
      }
      if (place_attached_object(object, cloud_from_root * root_from_link, primitives_, patches)) {
        ++totals_.payload_objects_filtered;
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "payload %s carries geometry this filter cannot place: it stays in "
                             "the cloud",
                             object.object_id.c_str());
      }
    }
  }

  void log_stats() {
    if (window_clouds_ == 0 && totals_.dropped_no_state == 0 && totals_.dropped_no_tf == 0) {
      return;
    }
    const double removed_pct =
        totals_.points_in > 0
            ? 100.0 * static_cast<double>(totals_.removed) / static_cast<double>(totals_.points_in)
            : 0.0;
    RCLCPP_INFO(this->get_logger(),
                "self-filter: %lu clouds published, %lu dropped (no joint state %lu, no TF %lu); "
                "%.1f%% of finite points removed; %.1f ms mean / %.1f ms max per cloud (last 5 s)",
                static_cast<unsigned long>(totals_.published),
                static_cast<unsigned long>(totals_.dropped_no_state + totals_.dropped_no_tf +
                                           totals_.dropped_bad_cloud),
                static_cast<unsigned long>(totals_.dropped_no_state),
                static_cast<unsigned long>(totals_.dropped_no_tf), removed_pct,
                window_clouds_ > 0 ? window_ms_sum_ / static_cast<double>(window_clouds_) : 0.0,
                window_ms_max_);
    window_clouds_ = 0;
    window_ms_sum_ = 0.0;
    window_ms_max_ = 0.0;
  }

  SelfModel model_;
  bool model_ok_{false};
  std::unordered_map<std::string, std::size_t> dof_of_name_;
  std::vector<std::size_t> required_dofs_;
  std::vector<double> q_;
  std::vector<std::uint8_t> seen_;
  std::deque<Snapshot> history_;

  double padding_m_{0.05};
  double max_skew_s_{0.1};
  double tf_timeout_s_{0.05};
  double history_s_{2.0};
  bool attached_enabled_{true};
  double attached_timeout_s_{0.5};

  std::vector<tf2::Transform> link_world_;
  std::vector<PayloadPrimitive> primitives_;
  SelfMask mask_;
  Totals totals_;
  std::uint64_t window_clouds_{0};
  double window_ms_sum_{0.0};
  double window_ms_max_{0.0};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  openral_msgs::msg::WorldStateStamped::SharedPtr world_state_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<openral_msgs::msg::WorldStateStamped>::SharedPtr world_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::TimerBase::SharedPtr stats_timer_;
};

}  // namespace openral_octomap_bridge

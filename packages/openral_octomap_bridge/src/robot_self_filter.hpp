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
// Joint positions are LINEARLY INTERPOLATED at the stamp between the two
// samples around it (`joint_state_at`). Every way of not knowing where the
// robot was when the cloud was captured — samples around the stamp more than
// two `joint_state_period_s` apart, no bracketing state within the bounded wait,
// nothing within `max_joint_state_skew_s`, a joint the model needs
// never reported, no camera transform at the stamp — DROPS THE CLOUD rather than forwarding it
// unfiltered or filtered against a guessed pose. A dropped cloud is silence downstream: the octree
// goes stale, the voxel bridge stops republishing (`max_octree_age_s`), and the kernel's world
// check fails closed. Forwarding it unfiltered would fill the map with the arm
// and stop the robot against itself; filtering it against a stale pose would
// punch the blind shell into the wrong place, which is the unsafe direction.
//
// A held payload is removed only with the same unique `tracked_object`
// binding the voxel bridge requires. Region, segmentation, fallback, stale,
// malformed, or ambiguous attachments leave their returns in the cloud: their
// geometry alone does not prove that those sensor points belong to the object.
//
// `cloud_out` carries x/y/z only, whatever the input carries: octomap reads
// nothing else. An input with more fields (a coloured sim cloud's `rgb`) also
// goes out whole, robot removed, on `cloud_out_rgb` for viewers — only while
// that topic has a subscriber, so a headless run never copies it.
//
// Each robot primitive's padding grows with its speed: `padding_m +
// |v|·velocity_padding_dt_s`, `|v|` the primitive's motion over the
// interpolated joint rate (`self_primitive_motion`), capped at
// `kMaxSelfFilterPaddingM`. It covers the stamp error interpolation cannot see.
//
// Second product, off by default (`robot_mask_enabled`): a robot mask per depth
// camera (`robot_mask_cameras`, with parallel `robot_mask_depth_topics`,
// `robot_mask_camera_info_topics`, `robot_mask_topics`). The UNPADDED model,
// posed at the depth image's stamp the same way, is rendered into the image
// (`render_robot_mask`) and published as mono8 (0 free, 128 uncertain, 255
// robot) with the DEPTH stamp. A frame the filter cannot pose publishes
// nothing: a refused frame is no evidence.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
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

#include "openral_octomap_bridge/identity_clearing.hpp"
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
    params.mimic_multiplier = declare_doubles("collision_mimic_multiplier");
    params.mimic_offset = declare_doubles("collision_mimic_offset");
    params.link_names = declare_strings("collision_link_names");
    params.capsule_link = declare_ints("collision_capsule_link");
    params.capsule_radius = declare_doubles("collision_capsule_radius");
    params.capsule_half_length = declare_doubles("collision_capsule_half_length");
    params.capsule_origin_xyzrpy = declare_doubles("collision_capsule_origin_xyzrpy");
    params.box_link = declare_ints("collision_box_link");
    params.box_half_extents = declare_doubles("collision_box_half_extents");
    params.box_origin_xyzrpy = declare_doubles("collision_box_origin_xyzrpy");
    // The kernel's tight geometry for those boxes (a 26-DOP and, within the
    // kernel's vertex budget, the exact convex hull). A box that carries one is
    // filtered against it rather than against the box's corner slack.
    params.box_hull = declare_ints("collision_box_hull");
    params.hull_dop_lo = declare_doubles("collision_hull_dop_lo");
    params.hull_dop_hi = declare_doubles("collision_hull_dop_hi");
    params.hull_vertex_first = declare_ints("collision_hull_vertex_first");
    params.hull_vertex_count = declare_ints("collision_hull_vertex_count");
    params.hull_vertices = declare_doubles("collision_hull_vertices");
    const auto joint_names = declare_strings("collision_joint_names");
    // Parallel to `collision_joint_names`: a second spelling each dof answers
    // to (the manifest's `sim_joint_name`, which is the upstream name a vendor
    // `/joint_states` uses). Empty entries are ignored.
    const auto joint_aliases = declare_strings("collision_joint_aliases");

    padding_m_ = this->declare_parameter<double>("padding_m", 0.02);
    max_skew_s_ = this->declare_parameter<double>("max_joint_state_skew_s", 0.1);
    // The joint-state stream's period: interpolation refuses a gap of more than
    // two of them and an extrapolation of more than one. Isaac simulation
    // launches start unready and receive the connected backend's actual tick
    // from the HAL over the standard ROS parameter service.
    period_from_hal_ = this->declare_parameter<bool>("joint_state_period_from_hal", false);
    period_s_ = this->declare_parameter<double>("joint_state_period_s", 0.05);
    period_ready_ = !period_from_hal_;
    // δt_sync: the stamp error left after interpolation (a camera's
    // exposure-to-stamp offset, the joint-state republish's stamp error). 0.02 s
    // adds 10 mm at 0.5 m/s; the padding cap still bounds the total.
    dt_sync_s_ = this->declare_parameter<double>("velocity_padding_dt_s", 0.02);
    // Issue #418 (new hazard row owed, Safety-WG review pending): the recent sweep. A
    // return within the padded volume the robot left over the last `trailing_sweep_s`
    // is the robot's own trailing / silhouette-edge return, not the world (Spark R16:
    // two finger-edge returns 15-21 mm outside the hull, landing 0.13 s after the
    // gripper left that cell, held a mid-air voxel that stopped it 2.4 s later).
    // 0 = off; the padding cap still bounds the total.
    trailing_sweep_s_ = this->declare_parameter<double>("trailing_sweep_s", 0.0);
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
    if ((!period_from_hal_ && !valid_period(period_s_)) || (period_from_hal_ && period_s_ != 0.0) ||
        !(dt_sync_s_ >= 0.0 && dt_sync_s_ <= 0.2) ||
        !(trailing_sweep_s_ >= 0.0 && trailing_sweep_s_ <= kMaxTrailingSweepS)) {
      RCLCPP_ERROR(this->get_logger(),
                   "joint_state_period_s=%g (0, 1] / velocity_padding_dt_s=%g [0, 0.2] / "
                   "trailing_sweep_s=%g [0, %g] out of range: forwarding NOTHING",
                   period_s_, dt_sync_s_, trailing_sweep_s_, kMaxTrailingSweepS);
      model_ok_ = false;
    }
    period_on_set_ = this->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = true;
          for (const auto& parameter : parameters) {
            if (parameter.get_name() != "joint_state_period_s") {
              continue;
            }
            if (!period_from_hal_ ||
                parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
                !valid_period(parameter.as_double())) {
              result.successful = false;
              result.reason = "joint_state_period_s requires joint_state_period_from_hal and "
                              "a finite value in (0, 1]";
              return result;
            }
          }
          return result;
        });
    period_post_set_ = this->add_post_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) {
          for (const auto& parameter : parameters) {
            if (parameter.get_name() == "joint_state_period_s") {
              period_s_ = parameter.as_double();
              // No sample from the old cadence may be used with the new rule.
              history_.clear();
              std::fill(q_.begin(), q_.end(), 0.0);
              std::fill(seen_.begin(), seen_.end(), 0);
              period_ready_ = true;
            }
          }
        });
    if (!valid_self_filter_padding(padding_m_)) {
      // Fail closed, as a bad model does: a wider shell hides obstacles the arm
      // can reach, and a negative or non-finite one is a misconfiguration.
      RCLCPP_ERROR(this->get_logger(),
                   "padding_m=%g is outside [0, %g] m: forwarding NOTHING (hazard log Entry 035)",
                   padding_m_, kMaxSelfFilterPaddingM);
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
    const auto hand = declare_strings("robot_mask_hand_links");
    const std::unordered_set<std::string> hand_set(hand.begin(), hand.end());
    for (const int link : model_.primitive_link) {
      mask_margin_.push_back(hand_set.count(model_.link_names[static_cast<std::size_t>(link)]) > 0
                                 ? kRobotMaskHandMarginM
                                 : kRobotMaskArmMarginM);
    }

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
    viewer_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_out_rgb",
                                                                        rclcpp::QoS(2).reliable());
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "cloud_in", rclcpp::SensorDataQoS().keep_last(2),
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { on_cloud(msg); });
    setup_robot_masks();
    sensor_retry_timer_ = this->create_wall_timer(std::chrono::milliseconds(5), [this]() {
      retry_clouds();
      for (std::size_t c = 0; c < mask_cameras_.size(); ++c) {
        retry_masks(c);
      }
    });
    stats_timer_ = this->create_wall_timer(std::chrono::seconds(5), [this]() { log_stats(); });

    const auto hulled = static_cast<std::size_t>(std::count_if(
        model_.primitive_hull.begin(), model_.primitive_hull.end(), [](int h) { return h >= 0; }));
    RCLCPP_INFO(this->get_logger(),
                "robot self-filter: %zu links, %zu primitives (%zu boxes refined by the "
                "kernel's hull), padding %.3f m, root %s, joint states %s (skew <= %.3f s)",
                model_.n_links(), model_.primitives.size(), hulled, padding_m_,
                model_.link_names.empty() ? "?" : model_.link_names.front().c_str(),
                joint_states_topic.c_str(), max_skew_s_);
  }

  /// Totals since start, for tests and the stats log.
  struct Totals {
    std::uint64_t clouds{0};
    std::uint64_t published{0};
    std::uint64_t dropped_no_state{0};
    std::uint64_t refused_joint_gap{0};      ///< samples around the stamp > 2 periods apart
    std::uint64_t refused_extrapolation{0};  ///< stamp > 1 period past the samples
    std::uint64_t dropped_no_tf{0};
    std::uint64_t dropped_bad_cloud{0};
    std::uint64_t points_in{0};
    std::uint64_t removed{0};
    std::uint64_t payload_objects_filtered{0};
    std::uint64_t masks_published{0};
    std::uint64_t masks_dropped{0};  ///< refused pose, no TF, no CameraInfo, bad image
  };
  const Totals& totals() const noexcept { return totals_; }

private:
  static bool valid_period(double value) {
    return std::isfinite(value) && value > 0.0 && value <= 1.0;
  }

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
    if (!period_ready_) {
      return;
    }
    const std::size_t n = std::min(msg->name.size(), msg->position.size());
    for (std::size_t i = 0; i < n; ++i) {
      const auto it = dof_of_name_.find(msg->name[i]);
      if (it == dof_of_name_.end() || it->second >= q_.size()) {
        continue;
      }
      q_[it->second] = msg->position[i];
      seen_[it->second] = 1;
    }
    if (!std::all_of(required_dofs_.begin(), required_dofs_.end(),
                     [this](std::size_t d) { return seen_[d] != 0; })) {
      return;  // an incomplete sample poses nothing
    }
    JointSample sample;
    sample.stamp_ns =
        rclcpp::Time(msg->header.stamp, this->get_clock()->get_clock_type()).nanoseconds();
    sample.q = q_;
    // Sorted by stamp: a late message lands where it belongs, not at the end.
    const auto at =
        std::upper_bound(history_.begin(), history_.end(), sample.stamp_ns,
                         [](std::int64_t t, const JointSample& s) { return t < s.stamp_ns; });
    history_.insert(at, std::move(sample));
    const auto horizon_ns = static_cast<std::int64_t>(history_s_ * 1e9);
    while (!history_.empty() && history_.back().stamp_ns - history_.front().stamp_ns > horizon_ns) {
      history_.pop_front();
    }
  }

  /// Pose the robot at `stamp` into `q_at_` / `qdot_at_`, or count and warn why not.
  bool joint_state_at_stamp(const rclcpp::Time& stamp, const char* what) {
    if (!period_ready_) {
      ++totals_.dropped_no_state;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "joint-state period has not been admitted by the HAL: dropping %s",
                           what);
      return false;
    }
    const auto verdict =
        joint_state_at(history_, stamp.nanoseconds(), period_s_, max_skew_s_, q_at_, qdot_at_);
    if (verdict == JointStateVerdict::kOk) {
      return true;
    }
    const char* why = "no complete joint state near";
    if (verdict == JointStateVerdict::kGap) {
      ++totals_.refused_joint_gap;
      why = "joint states more than 2 periods apart around";
    } else if (verdict == JointStateVerdict::kExtrapolation) {
      ++totals_.refused_extrapolation;
      why = "more than 1 period of joint-state extrapolation to";
    } else {
      ++totals_.dropped_no_state;
    }
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "%s the %s stamp (period %.3f s, history %zu): dropping it", why, what,
                         period_s_, history_.size());
    return false;
  }

  static std::vector<sensor_msgs::msg::PointField> xyz_fields() {
    std::vector<sensor_msgs::msg::PointField> fields(3);
    for (std::uint32_t i = 0; i < 3; ++i) {
      fields[i].name = std::string(1, static_cast<char>('x' + i));
      fields[i].offset = 4 * i;
      fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
      fields[i].count = 1;
    }
    return fields;
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

  bool state_covers(const builtin_interfaces::msg::Time& stamp) const {
    const auto ns = rclcpp::Time(stamp, this->get_clock()->get_clock_type()).nanoseconds();
    return !history_.empty() && history_.front().stamp_ns <= ns && history_.back().stamp_ns >= ns;
  }

  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    ++totals_.clouds;
    if (!model_ok_) {
      return;
    }
    retry_clouds();
    if (state_covers(msg->header.stamp)) {
      filter_cloud(msg);
      return;
    }
    if (!std::isfinite(max_skew_s_) || max_skew_s_ <= 0.0 ||
        msg->data.size() > 32U * 1024U * 1024U) {
      ++totals_.dropped_no_state;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "cloud: matching joint state unavailable or buffer exceeds bounded "
                           "storage: refusing cloud");
      return;
    }
    if (pending_clouds_.size() == 2) {
      pending_clouds_.pop_front();
      ++totals_.dropped_no_state;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "cloud: pending joint-state queue full: oldest cloud refused");
    }
    pending_clouds_.push_back({msg, std::chrono::steady_clock::now()});
  }

  void retry_clouds() {
    for (auto it = pending_clouds_.begin(); it != pending_clouds_.end();) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - it->received).count() >=
          max_skew_s_) {
        ++totals_.dropped_no_state;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "cloud: capture-time joint-state wait expired: refusing cloud");
        it = pending_clouds_.erase(it);
      } else if (state_covers(it->cloud->header.stamp)) {
        filter_cloud(it->cloud);
        it = pending_clouds_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void filter_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    const auto t0 = std::chrono::steady_clock::now();
    const rclcpp::Time stamp(msg->header.stamp, this->get_clock()->get_clock_type());

    if (!joint_state_at_stamp(stamp, "cloud")) {
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

    self_forward_kinematics(model_, q_at_.data(), q_at_.size(), link_world_);
    // Where the joint rate carries each primitive within δt_sync: its padding.
    q_later_.resize(q_at_.size());
    for (std::size_t d = 0; d < q_at_.size(); ++d) {
      q_later_[d] = q_at_[d] + qdot_at_[d] * dt_sync_s_;
    }
    self_forward_kinematics(model_, q_later_.data(), q_later_.size(), link_later_);
    self_primitive_motion(model_, link_world_, link_later_, motion_);
    if (trailing_sweep_s_ > 0.0) {
      // The pose `trailing_sweep_s` ago, from the same history: each primitive is
      // also padded by how far it has moved since. No such state (a fresh start)
      // = today's padding, which keeps robot points (a false stop), never erases world.
      const std::int64_t before_ns =
          stamp.nanoseconds() - static_cast<std::int64_t>(trailing_sweep_s_ * 1e9);
      if (joint_state_at(history_, before_ns, period_s_, max_skew_s_, q_before_, qdot_before_) ==
          JointStateVerdict::kOk) {
        self_forward_kinematics(model_, q_before_.data(), q_before_.size(), link_before_);
        self_primitive_motion(model_, link_world_, link_before_, motion_before_);
        for (std::size_t k = 0; k < motion_.size() && k < motion_before_.size(); ++k) {
          motion_[k] = std::max(motion_[k], motion_before_[k]);
        }
      }
    }
    primitives_.clear();
    primitive_hulls_.clear();
    place_self_primitives(model_, link_world_, cloud_from_root, primitives_, &primitive_hulls_);
    add_payload_primitives(stamp, cloud_from_root);                 // appended past the hulls: none
    mask_.set(primitives_, padding_m_, primitive_hulls_, motion_);  // payloads: base padding

    std::size_t xo = 0;
    std::size_t yo = 0;
    std::size_t zo = 0;
    const std::size_t n_points = static_cast<std::size_t>(msg->width) * msg->height;
    if (!float_field(*msg, "x", xo) || !float_field(*msg, "y", yo) || !float_field(*msg, "z", zo) ||
        msg->point_step < 12 ||
        !xyz_cloud_layout_valid(msg->data.size(), n_points, msg->point_step, xo, yo, zo)) {
      ++totals_.dropped_bad_cloud;
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 5000,
          "cloud has no FLOAT32 x/y/z, a coordinate outside its record or a short "
          "data buffer: dropping it");
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
    if (msg->fields.size() > 3 || msg->point_step != 12) {
      if (viewer_pub_->get_subscription_count() > 0) {
        viewer_pub_->publish(*out);  // the whole records, colour and all
      }
      out->data = xyz_records(out->data, out->point_step, xo, yo, zo);
      out->fields = xyz_fields();
      out->point_step = 12;
      out->row_step = out->point_step * out->width;
    }
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
      const char* why = "no_robot_root";
      if (model_.link_names.empty() ||
          identity_clearing_track(object, *state, model_.link_names.front(), &why) == nullptr) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "payload %s evidence '%s' has no unique valid tracked-object "
                             "binding in the robot root frame (reason=%s); its sensor returns "
                             "stay in the cloud",
                             object.object_id.c_str(), object.evidence_kind.c_str(), why);
        continue;
      }
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

  struct PendingMask {
    sensor_msgs::msg::Image::ConstSharedPtr image;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    std::chrono::steady_clock::time_point received;
  };

  struct MaskCamera {
    std::string name;
    std::deque<PendingMask> pending;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub;
  };

  void setup_robot_masks() {
    const bool enabled = this->declare_parameter<bool>("robot_mask_enabled", false);
    const auto names = declare_strings("robot_mask_cameras");
    const auto depth = declare_strings("robot_mask_depth_topics");
    const auto info = declare_strings("robot_mask_camera_info_topics");
    const auto out = declare_strings("robot_mask_topics");
    mask_uncertain_px_ =
        static_cast<int>(this->declare_parameter<std::int64_t>("robot_mask_uncertain_px", 3));
    if (!enabled || !model_ok_) {
      return;
    }
    if (depth.size() != names.size() || info.size() != names.size() || out.size() != names.size() ||
        names.size() > 16 || !std::isfinite(max_skew_s_) || max_skew_s_ < 0.0 ||
        mask_uncertain_px_ < 0 || mask_uncertain_px_ > 64) {
      RCLCPP_ERROR(this->get_logger(),
                   "robot_mask_* arrays disagree in length or robot_mask_uncertain_px is outside "
                   "[0, 64], more than 16 cameras, or invalid skew: publishing NO robot mask");
      return;
    }
    mask_cameras_.resize(names.size());
    for (std::size_t c = 0; c < names.size(); ++c) {
      MaskCamera& cam = mask_cameras_[c];
      cam.name = names[c];
      // BEST_EFFORT, VOLATILE, KEEP_LAST 5: a sensor-class stream (CLAUDE.md §2).
      cam.pub =
          this->create_publisher<sensor_msgs::msg::Image>(out[c], rclcpp::QoS(5).best_effort());
      cam.info_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>(
          info[c], rclcpp::SensorDataQoS().keep_last(1),
          [this, c](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
            mask_cameras_[c].info = std::move(msg);
          });
      cam.depth_sub = this->create_subscription<sensor_msgs::msg::Image>(
          depth[c], rclcpp::SensorDataQoS().keep_last(2),
          [this, c](sensor_msgs::msg::Image::ConstSharedPtr msg) { on_depth(c, std::move(msg)); });
      RCLCPP_INFO(this->get_logger(), "robot mask: camera %s, depth %s -> %s", names[c].c_str(),
                  depth[c].c_str(), out[c].c_str());
    }
  }

  /// The depth image as metres into `depth_m_`; false on an encoding or a
  /// buffer this node cannot read.
  bool depth_metres(const sensor_msgs::msg::Image& img) {
    const std::size_t w = img.width;
    const std::size_t h = img.height;
    const bool f32 = img.encoding == "32FC1";
    const bool u16 = img.encoding == "16UC1" || img.encoding == "mono16";
    const std::size_t bpp = f32 ? 4 : 2;
    if ((!f32 && !u16) || img.step < w * bpp || img.data.size() < img.step * h) {
      return false;
    }
    depth_m_.resize(w * h);
    for (std::size_t v = 0; v < h; ++v) {
      const std::uint8_t* row = img.data.data() + v * img.step;
      for (std::size_t u = 0; u < w; ++u) {
        if (f32) {
          std::memcpy(&depth_m_[v * w + u], row + 4 * u, 4);
        } else {
          std::uint16_t mm = 0;
          std::memcpy(&mm, row + 2 * u, 2);
          depth_m_[v * w + u] =
              mm == 0 ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(mm) * 1e-3F;
        }
      }
    }
    return true;
  }

  void on_depth(std::size_t c, sensor_msgs::msg::Image::ConstSharedPtr image) {
    MaskCamera& cam = mask_cameras_[c];
    // Bound retained sensor buffers as well as their count (up to 4K float depth).
    if (image->data.size() > 32U * 1024U * 1024U ||
        static_cast<std::uint64_t>(image->width) * image->height > 8U * 1024U * 1024U) {
      ++totals_.masks_dropped;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "robot mask %s: depth exceeds bounded image storage: no mask",
                           cam.name.c_str());
      return;
    }
    retry_masks(c);
    if (publish_mask(c, *image, cam.info)) {
      return;
    }
    if (max_skew_s_ == 0.0) {
      ++totals_.masks_dropped;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "robot mask %s: capture-time TF/joint state unavailable and no wait "
                           "allowed: no mask",
                           cam.name.c_str());
      return;
    }
    if (cam.pending.size() == 2) {
      cam.pending.pop_front();
      ++totals_.masks_dropped;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "robot mask %s: pending sensor queue full: oldest depth refused",
                           cam.name.c_str());
    }
    cam.pending.push_back({std::move(image), cam.info, std::chrono::steady_clock::now()});
  }

  void retry_masks(std::size_t c) {
    auto& pending = mask_cameras_[c].pending;
    for (auto it = pending.begin(); it != pending.end();) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - it->received).count() >=
          max_skew_s_) {
        ++totals_.masks_dropped;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "robot mask %s: capture-time TF/joint-state wait expired: no mask",
                             mask_cameras_[c].name.c_str());
        it = pending.erase(it);
      } else if (publish_mask(c, *it->image, it->info)) {
        it = pending.erase(it);
      } else {
        ++it;
      }
    }
  }

  // False means capture-time camera TF or a bracketing joint state is still pending.
  bool publish_mask(std::size_t c, const sensor_msgs::msg::Image& img,
                    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info) {
    MaskCamera& cam = mask_cameras_[c];
    const rclcpp::Time stamp(img.header.stamp, this->get_clock()->get_clock_type());
    if (stamp.nanoseconds() <= 0 || info == nullptr || info->width != img.width ||
        info->height != img.height || img.width == 0 || img.height == 0) {
      ++totals_.masks_dropped;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "robot mask %s: no matching CameraInfo or an unreadable depth image "
                           "(%s): no mask",
                           cam.name.c_str(), img.encoding.c_str());
      return true;
    }
    if (!state_covers(img.header.stamp)) {
      return false;
    }
    tf2::Transform cam_from_root;
    try {
      const auto tf_msg =
          tf_buffer_->lookupTransform(img.header.frame_id, model_.link_names.front(), stamp,
                                      rclcpp::Duration::from_seconds(0.0));
      tf2::fromMsg(tf_msg.transform, cam_from_root);
    } catch (const tf2::TransformException&) {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "robot mask %s: waiting at most %g s for exact-stamp camera TF",
                            cam.name.c_str(), max_skew_s_);
      return false;
    }
    if (!depth_metres(img)) {
      ++totals_.masks_dropped;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "robot mask %s: unreadable depth image: no mask", cam.name.c_str());
      return true;
    }
    if (!joint_state_at_stamp(stamp, "depth image")) {
      ++totals_.masks_dropped;
      return true;
    }
    // The UNPADDED model at the depth stamp; never a payload (it is an object).
    self_forward_kinematics(model_, q_at_.data(), q_at_.size(), mask_link_world_);
    mask_primitives_.clear();
    mask_hulls_.clear();
    place_self_primitives(model_, mask_link_world_, cam_from_root, mask_primitives_, &mask_hulls_);
    PinholeCamera pinhole;
    pinhole.fx = info->k[0];
    pinhole.cx = info->k[2];
    pinhole.fy = info->k[4];
    pinhole.cy = info->k[5];
    pinhole.width = img.width;
    pinhole.height = img.height;
    auto out = std::make_unique<sensor_msgs::msg::Image>();
    out->header = img.header;  // the DEPTH stamp and frame
    out->width = img.width;
    out->height = img.height;
    out->encoding = "mono8";
    out->step = img.width;
    render_robot_mask(mask_primitives_, mask_hulls_, mask_margin_, depth_m_.data(), pinhole,
                      mask_uncertain_px_, mask_scratch_, out->data);
    cam.pub->publish(std::move(out));
    ++totals_.masks_published;
    return true;
  }

  void log_stats() {
    if (window_clouds_ == 0 && totals_.dropped_no_state == 0 && totals_.dropped_no_tf == 0 &&
        totals_.masks_published == 0 && totals_.masks_dropped == 0) {
      return;
    }
    const double removed_pct =
        totals_.points_in > 0
            ? 100.0 * static_cast<double>(totals_.removed) / static_cast<double>(totals_.points_in)
            : 0.0;
    RCLCPP_INFO(this->get_logger(),
                "self-filter: %lu clouds published, %lu dropped (no joint state %lu, joint gap "
                "%lu, extrapolation %lu, no TF %lu); %lu robot masks (%lu dropped); "
                "%.1f%% of finite points removed; %.1f ms mean / %.1f ms max per cloud (last 5 s)",
                static_cast<unsigned long>(totals_.published),
                static_cast<unsigned long>(totals_.dropped_no_state + totals_.refused_joint_gap +
                                           totals_.refused_extrapolation + totals_.dropped_no_tf +
                                           totals_.dropped_bad_cloud),
                static_cast<unsigned long>(totals_.dropped_no_state),
                static_cast<unsigned long>(totals_.refused_joint_gap),
                static_cast<unsigned long>(totals_.refused_extrapolation),
                static_cast<unsigned long>(totals_.dropped_no_tf),
                static_cast<unsigned long>(totals_.masks_published),
                static_cast<unsigned long>(totals_.masks_dropped), removed_pct,
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
  std::deque<JointSample> history_;
  std::vector<double> q_at_;     ///< the last interpolated pose
  std::vector<double> qdot_at_;  ///< and its joint rate
  std::vector<double> q_later_;

  double padding_m_{0.02};
  double max_skew_s_{0.1};
  double period_s_{0.05};
  bool period_from_hal_{false};
  bool period_ready_{true};
  double dt_sync_s_{0.02};
  double tf_timeout_s_{0.05};
  double history_s_{2.0};
  bool attached_enabled_{true};
  double attached_timeout_s_{0.5};

  std::vector<tf2::Transform> link_world_;
  std::vector<tf2::Transform> link_later_;
  std::vector<double> motion_;  ///< per robot primitive: its motion within δt_sync
  double trailing_sweep_s_{0.0};
  std::vector<double> q_before_;
  std::vector<double> qdot_before_;
  std::vector<tf2::Transform> link_before_;
  std::vector<double> motion_before_;  ///< per robot primitive: its motion since `trailing_sweep_s`
  std::vector<PayloadPrimitive> primitives_;
  std::vector<const SelfHull*> primitive_hulls_;  ///< parallel to the robot's primitives
  SelfMask mask_;
  struct PendingCloud {
    sensor_msgs::msg::PointCloud2::SharedPtr cloud;
    std::chrono::steady_clock::time_point received;
  };
  std::deque<PendingCloud> pending_clouds_;
  // Robot mask (`robot_mask_enabled`).
  std::vector<MaskCamera> mask_cameras_;
  std::vector<double> mask_margin_;  ///< per robot primitive: its backstop margin
  int mask_uncertain_px_{3};
  std::vector<tf2::Transform> mask_link_world_;
  std::vector<PayloadPrimitive> mask_primitives_;
  std::vector<const SelfHull*> mask_hulls_;
  std::vector<float> depth_m_;
  RobotMaskScratch mask_scratch_;
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
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr viewer_pub_;
  rclcpp::TimerBase::SharedPtr stats_timer_;
  rclcpp::TimerBase::SharedPtr sensor_retry_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr period_on_set_;
  rclcpp::node_interfaces::PostSetParametersCallbackHandle::SharedPtr period_post_set_;
};

}  // namespace openral_octomap_bridge

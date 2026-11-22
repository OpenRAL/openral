// SPDX-License-Identifier: Apache-2.0
// Peer producer (ADR-0137, hazard log Entry 103): every other robot of the
// workcell, from its own geometry, as `/openral/peer_voxels` for this robot's
// safety kernel.
//
// Each peer's graph is on its own ROS domain. This node joins each peer's domain
// with a context of its own and reads only the peer's `/joint_states` and `/tf`
// (its `world -> map -> odom -> base` chain); it never publishes there. The peer's
// collision model comes in as parameters under `peer_<name>.`, lowered by this
// robot's launch from the peer's manifest exactly as the peer's own kernel gets it.
//
// Fail closed: while any peer's joints or base pose, or this robot's own base
// pose, is missing or older than `max_input_age_s`, nothing is published, and the
// kernel's receipt deadline refuses every chunk. A peer is never assumed gone.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <openral_msgs/msg/occupancy_voxels.hpp>
#include <rclcpp/rclcpp.hpp>

#include "openral_octomap_bridge/peer_grid.hpp"
#include "openral_octomap_bridge/self_filter.hpp"

namespace openral_octomap_bridge {

class PeerVoxels : public rclcpp::Node {
public:
  explicit PeerVoxels(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : rclcpp::Node("openral_peer_voxels", options) {
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    centre_ = tf2::Vector3(declare_parameter<double>("coverage_center_x", 0.0),
                           declare_parameter<double>("coverage_center_y", 0.0),
                           declare_parameter<double>("coverage_center_z", 0.5));
    radius_ = declare_parameter<double>("coverage_radius_m", 1.05);
    resolution_ = declare_parameter<double>("resolution", 0.025);
    horizon_s_ = declare_parameter<double>("sweep_horizon_s", 0.3);
    max_age_s_ = declare_parameter<double>("max_input_age_s", 0.3);
    pad_m_ = declare_parameter<double>("pad_m", 0.02);
    const double period_s = declare_parameter<double>("publish_period_s", 0.05);
    const auto names = declare_parameter<std::vector<std::string>>("peer_names", {});
    const auto domains = declare_parameter<std::vector<std::int64_t>>("peer_domains", {});
    const bool sim_time = get_parameter("use_sim_time").as_bool();

    pub_ = create_publisher<openral_msgs::msg::OccupancyVoxels>(
        "/openral/peer_voxels", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, false);

    if (names.size() != domains.size() || names.empty() || !(resolution_ > 0.0) ||
        !(radius_ > 0.0) || !(horizon_s_ >= max_age_s_) || !(max_age_s_ > 0.0)) {
      RCLCPP_ERROR(get_logger(),
                   "peer_voxels: peer_names/peer_domains disagree or are empty, or the grid or "
                   "sweep is invalid (sweep_horizon_s must cover max_input_age_s): publishing "
                   "NOTHING");
      return;
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
      auto peer = std::make_unique<Peer>();
      peer->name = names[i];
      if (!load_peer(*peer)) {
        return;  // fail closed: one unusable peer and nothing is published
      }
      join_domain(*peer, static_cast<std::size_t>(domains[i]), sim_time);
      peers_.push_back(std::move(peer));
    }
    timer_ = create_wall_timer(std::chrono::duration<double>(period_s), [this] { publish(); });
    RCLCPP_INFO(get_logger(), "peer_voxels: %zu peer(s) into %s, grid %d^3 at %.3f m",
                peers_.size(), base_frame_.c_str(), peer_cells_per_axis(radius_, resolution_),
                resolution_);
  }

  ~PeerVoxels() override {
    for (auto& peer : peers_) {
      if (peer->executor) {
        peer->executor->cancel();
      }
      if (peer->spin.joinable()) {
        peer->spin.join();
      }
      if (peer->context) {
        peer->context->shutdown("peer_voxels exit");
      }
    }
  }

  PeerVoxels(const PeerVoxels&) = delete;
  PeerVoxels& operator=(const PeerVoxels&) = delete;

private:
  struct Peer {
    std::string name;
    SelfModel model;
    std::vector<std::size_t> base_dofs;
    std::string base_frame;
    std::unordered_map<std::string, std::size_t> dof_of;
    std::size_t n_dof{0};
    // Latest measured joints (guarded by `mutex`).
    std::mutex mutex;
    std::vector<double> q;
    std::vector<double> dq;
    std::vector<double> q_prev;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    rclcpp::Time stamp_prev{0, 0, RCL_ROS_TIME};
    bool have_q{false};
    // Its domain.
    rclcpp::Context::SharedPtr context;
    rclcpp::Node::SharedPtr node;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
    std::thread spin;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  };

  bool load_peer(Peer& peer) {
    const std::string p = "peer_" + peer.name + ".";
    const auto ints = [&](const char* k) {
      return declare_parameter<std::vector<std::int64_t>>(p + k, std::vector<std::int64_t>{});
    };
    const auto doubles = [&](const char* k) {
      return declare_parameter<std::vector<double>>(p + k, std::vector<double>{});
    };
    const auto strings = [&](const char* k) {
      return declare_parameter<std::vector<std::string>>(p + k, std::vector<std::string>{});
    };
    SelfModelParams params;
    params.n_links = declare_parameter<std::int64_t>(p + "collision_n_links", 0);
    params.parent = ints("collision_parent");
    params.joint_kind = ints("collision_joint_kind");
    params.dof_index = ints("collision_dof_index");
    params.origin_xyzrpy = doubles("collision_origin_xyzrpy");
    params.axis = doubles("collision_axis");
    params.mimic_multiplier = doubles("collision_mimic_multiplier");
    params.mimic_offset = doubles("collision_mimic_offset");
    params.curve_point_first = ints("collision_curve_point_first");
    params.curve_point_count = ints("collision_curve_point_count");
    params.curve_points = doubles("collision_curve_points");
    params.link_names = strings("collision_link_names");
    params.capsule_link = ints("collision_capsule_link");
    params.capsule_radius = doubles("collision_capsule_radius");
    params.capsule_half_length = doubles("collision_capsule_half_length");
    params.capsule_origin_xyzrpy = doubles("collision_capsule_origin_xyzrpy");
    params.box_link = ints("collision_box_link");
    params.box_half_extents = doubles("collision_box_half_extents");
    params.box_origin_xyzrpy = doubles("collision_box_origin_xyzrpy");
    // Peers are posed from their boxes, not tight hulls: a box contains its hull,
    // so this only ever marks more cells.
    const auto joint_names = strings("collision_joint_names");
    const auto aliases = strings("collision_joint_aliases");
    for (const auto d : ints("collision_base_dofs")) {
      peer.base_dofs.push_back(static_cast<std::size_t>(d));
    }
    peer.base_frame = declare_parameter<std::string>(p + "base_frame", "base_link");
    std::string error;
    if (!build_self_model(params, peer.model, error) ||
        joint_names.size() < peer.model.required_dof) {
      RCLCPP_ERROR(get_logger(),
                   "peer_voxels: peer %s has no usable collision model (%s): "
                   "publishing NOTHING",
                   peer.name.c_str(),
                   error.empty() ? "too few collision_joint_names" : error.c_str());
      return false;
    }
    peer.n_dof = joint_names.size();
    for (std::size_t d = 0; d < joint_names.size(); ++d) {
      peer.dof_of[joint_names[d]] = d;
      if (d < aliases.size() && !aliases[d].empty()) {
        peer.dof_of[aliases[d]] = d;
      }
    }
    peer.q.assign(peer.n_dof, 0.0);
    peer.dq.assign(peer.n_dof, 0.0);
    peer.q_prev.assign(peer.n_dof, 0.0);
    return true;
  }

  void join_domain(Peer& peer, std::size_t domain, bool sim_time) {
    rclcpp::InitOptions init;
    init.set_domain_id(domain);
    init.auto_initialize_logging(false);
    peer.context = std::make_shared<rclcpp::Context>();
    peer.context->init(0, nullptr, init);
    rclcpp::NodeOptions opts;
    opts.context(peer.context);
    opts.parameter_overrides({rclcpp::Parameter("use_sim_time", sim_time)});
    opts.start_parameter_services(false);
    opts.start_parameter_event_publisher(false);
    peer.node = std::make_shared<rclcpp::Node>("openral_peer_voxels_reader", opts);
    peer.tf_buffer = std::make_shared<tf2_ros::Buffer>(peer.node->get_clock());
    peer.tf_listener =
        std::make_shared<tf2_ros::TransformListener>(*peer.tf_buffer, peer.node, false);
    Peer* raw = &peer;
    peer.js_sub = peer.node->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::SensorDataQoS(),
        [this, raw](const sensor_msgs::msg::JointState::SharedPtr msg) { on_joints(*raw, *msg); });
    rclcpp::ExecutorOptions eopts;
    eopts.context = peer.context;
    peer.executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>(eopts);
    peer.executor->add_node(peer.node);
    peer.spin = std::thread([exec = peer.executor] { exec->spin(); });
  }

  void on_joints(Peer& peer, const sensor_msgs::msg::JointState& msg) {
    const rclcpp::Time stamp = (msg.header.stamp.sec != 0 || msg.header.stamp.nanosec != 0)
                                   ? rclcpp::Time(msg.header.stamp, RCL_ROS_TIME)
                                   : peer.node->now();
    std::lock_guard<std::mutex> lock(peer.mutex);
    if (peer.have_q) {
      peer.q_prev = peer.q;
      peer.stamp_prev = peer.stamp;
    }
    bool has_velocity = msg.velocity.size() == msg.name.size();
    for (std::size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i) {
      const auto it = peer.dof_of.find(msg.name[i]);
      if (it == peer.dof_of.end() || !std::isfinite(msg.position[i])) {
        continue;
      }
      peer.q[it->second] = msg.position[i];
      if (has_velocity && std::isfinite(msg.velocity[i])) {
        peer.dq[it->second] = msg.velocity[i];
      }
    }
    // No measured velocity: the change since the previous sample.
    const double dt = (stamp - peer.stamp_prev).seconds();
    if (!has_velocity && peer.have_q && dt > 0.0) {
      for (std::size_t d = 0; d < peer.n_dof; ++d) {
        peer.dq[d] = (peer.q[d] - peer.q_prev[d]) / dt;
      }
    }
    peer.stamp = stamp;
    peer.have_q = true;
  }

  // A transform and its stamp, or false.
  static bool lookup(tf2_ros::Buffer& buffer, const std::string& target, const std::string& source,
                     const rclcpp::Time& at, tf2::Transform& out, rclcpp::Time& stamp) {
    try {
      const auto msg = buffer.lookupTransform(target, source, tf2_ros::fromRclcpp(at));
      tf2::fromMsg(msg.transform, out);
      stamp = rclcpp::Time(msg.header.stamp, RCL_ROS_TIME);
      return true;
    } catch (const tf2::TransformException&) {
      return false;
    }
  }

  void refuse(const char* why, const std::string& who) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "peer_voxels: %s (%s): publishing nothing, the kernel refuses", why,
                         who.c_str());
  }

  void publish() {
    if (peers_.empty()) {
      return;
    }
    const rclcpp::Time now = this->now();
    const auto too_old = [&](const rclcpp::Time& t) { return (now - t).seconds() > max_age_s_; };
    tf2::Transform own_from_world;
    rclcpp::Time oldest = now;
    rclcpp::Time own_stamp;
    if (!lookup(*tf_buffer_, base_frame_, world_frame_, rclcpp::Time(0, 0, RCL_ROS_TIME),
                own_from_world, own_stamp)) {
      return refuse("no own base pose in world", base_frame_);
    }
    // A static-only chain (a fixed arm) is stamped 0: it never ages.
    if (own_stamp.nanoseconds() != 0) {
      if (too_old(own_stamp)) {
        return refuse("own base pose is stale", base_frame_);
      }
      oldest = std::min(oldest, own_stamp);
    }
    PeerGrid grid = empty_peer_grid(centre_, radius_, resolution_);
    for (auto& peer : peers_) {
      std::vector<double> q;
      std::vector<double> dq;
      rclcpp::Time js_stamp;
      {
        std::lock_guard<std::mutex> lock(peer->mutex);
        if (!peer->have_q) {
          return refuse("no joint states yet", peer->name);
        }
        q = peer->q;
        dq = peer->dq;
        js_stamp = peer->stamp;
      }
      if (too_old(js_stamp)) {
        return refuse("joint states are stale", peer->name);
      }
      oldest = std::min(oldest, js_stamp);
      tf2::Transform base_now;
      rclcpp::Time base_stamp;
      if (!lookup(*peer->tf_buffer, world_frame_, peer->base_frame,
                  rclcpp::Time(0, 0, RCL_ROS_TIME), base_now, base_stamp)) {
        return refuse("no base pose in world", peer->name);
      }
      PlanarTwist twist;
      if (base_stamp.nanoseconds() != 0) {
        if (too_old(base_stamp)) {
          return refuse("base pose is stale", peer->name);
        }
        oldest = std::min(oldest, base_stamp);
        tf2::Transform base_before;
        rclcpp::Time before_stamp;
        const rclcpp::Time before = base_stamp - rclcpp::Duration::from_seconds(kTwistWindowS);
        if (!lookup(*peer->tf_buffer, world_frame_, peer->base_frame, before, base_before,
                    before_stamp)) {
          return refuse("no base motion history", peer->name);
        }
        twist = planar_twist(base_before, base_now, kTwistWindowS);
      }
      std::vector<PayloadPrimitive> prims;
      peer_swept_primitives(peer->model, q, dq, peer->base_dofs, own_from_world, base_now, twist,
                            horizon_s_, prims);
      rasterize_primitives(prims, pad_m_, grid);
    }
    openral_msgs::msg::OccupancyVoxels msg;
    msg.header.stamp = now;
    msg.header.frame_id = base_frame_;
    msg.source_stamp = oldest;
    msg.origin.x = grid.origin.x();
    msg.origin.y = grid.origin.y();
    msg.origin.z = grid.origin.z();
    msg.orientation.w = 1.0;
    msg.resolution = grid.resolution;
    msg.size_x = msg.size_y = msg.size_z = static_cast<std::uint32_t>(grid.n);
    msg.occupancy = std::move(grid.occupancy);
    pub_->publish(msg);
  }

  /// How far back a peer's base pose is read to measure its motion, s.
  static constexpr double kTwistWindowS = 0.1;

  std::string base_frame_;
  std::string world_frame_;
  tf2::Vector3 centre_;
  double radius_{0.0};
  double resolution_{0.0};
  double horizon_s_{0.0};
  double max_age_s_{0.0};
  double pad_m_{0.0};
  std::vector<std::unique_ptr<Peer>> peers_;
  rclcpp::Publisher<openral_msgs::msg::OccupancyVoxels>::SharedPtr pub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace openral_octomap_bridge

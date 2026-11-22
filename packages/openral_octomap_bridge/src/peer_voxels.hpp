// SPDX-License-Identifier: Apache-2.0
// Peer producer (ADR-0137, hazard log Entry 103): every other robot of the
// workcell, from its own geometry, as `/openral/peer_voxels` for this robot's
// safety kernel.
//
// Each peer's graph is on its own ROS domain. This node joins each peer's domain
// with a context of its own and reads only the peer's `/joint_states` and `/tf`
// (its `world -> map -> odom -> base` chain); it never publishes there. The peer's
// collision model comes in as parameters under `peer_<name>.`, lowered by this
// robot's launch from the peer's manifest.
//
// Fail closed: while any peer's joints (each joint on its own stamp) or base pose, or
// this robot's own base pose, is missing, older than `max_input_age_s` or stamped
// further ahead than that, nothing is published, and the kernel's receipt deadline
// refuses every chunk. A peer is never assumed gone.

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
#include "self_model_params.hpp"

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
    // The launch sets both from the kernel's peer data-age budget: the sweep must
    // cover at least as long as the kernel trusts a grid.
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
    grid_ = empty_peer_grid(centre_, radius_, resolution_);
    for (std::size_t i = 0; i < names.size(); ++i) {
      auto peer = std::make_unique<Peer>();
      peer->name = names[i];
      if (!load_peer(*peer)) {
        for (auto& joined : peers_) {
          stop(*joined);
        }
        peers_.clear();
        return;  // fail closed: one unusable peer and nothing is published
      }
      join_domain(*peer, static_cast<std::size_t>(domains[i]), sim_time);
      peers_.push_back(std::move(peer));
    }
    timer_ = create_wall_timer(std::chrono::duration<double>(period_s), [this] { publish(); });
    RCLCPP_INFO(get_logger(), "peer_voxels: %zu peer(s) into %s, grid %d^3 at %.3f m",
                peers_.size(), base_frame_.c_str(), grid_.n, resolution_);
  }

  ~PeerVoxels() override {
    for (auto& peer : peers_) {
      stop(*peer);
    }
  }

  PeerVoxels(const PeerVoxels&) = delete;
  PeerVoxels& operator=(const PeerVoxels&) = delete;

private:
  struct Peer {
    std::string name;
    SelfModel model;
    std::vector<std::size_t> base_dofs;
    /// The dofs its pose depends on: every dof a link uses, except base dofs.
    std::vector<std::size_t> required;
    std::string base_frame;
    std::unordered_map<std::string, std::size_t> dof_of;
    std::mutex mutex;
    std::unique_ptr<PeerJoints> joints;  ///< guarded by `mutex`
    // Its domain.
    rclcpp::Context::SharedPtr context;
    rclcpp::Node::SharedPtr node;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
    std::thread spin;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  };

  static void stop(Peer& peer) {
    if (peer.executor) {
      peer.executor->cancel();
    }
    if (peer.spin.joinable()) {
      peer.spin.join();
    }
    if (peer.context) {
      peer.context->shutdown("peer_voxels exit");
    }
  }

  bool load_peer(Peer& peer) {
    const std::string p = "peer_" + peer.name + ".";
    DeclaredSelfModel declared = declare_self_model(*this, p);
    // Peers are posed from their boxes, not tight hulls: a box contains its hull, so
    // this only ever marks more cells.
    declared.params.box_hull.clear();
    declared.params.hull_dop_lo.clear();
    declared.params.hull_dop_hi.clear();
    declared.params.hull_vertex_first.clear();
    declared.params.hull_vertex_count.clear();
    declared.params.hull_vertices.clear();
    for (const auto d : declare_parameter<std::vector<std::int64_t>>(p + "collision_base_dofs",
                                                                     std::vector<std::int64_t>{})) {
      peer.base_dofs.push_back(static_cast<std::size_t>(d));
    }
    peer.base_frame = declare_parameter<std::string>(p + "base_frame", "base_link");
    std::string error;
    const auto& joint_names = declared.joint_names;
    if (!build_self_model(declared.params, peer.model, error) ||
        joint_names.size() < peer.model.required_dof) {
      RCLCPP_ERROR(get_logger(),
                   "peer_voxels: peer %s has no usable collision model (%s): publishing NOTHING",
                   peer.name.c_str(),
                   error.empty() ? "too few collision_joint_names" : error.c_str());
      return false;
    }
    for (std::size_t d = 0; d < joint_names.size(); ++d) {
      peer.dof_of[joint_names[d]] = d;
      if (d < declared.joint_aliases.size() && !declared.joint_aliases[d].empty()) {
        peer.dof_of[declared.joint_aliases[d]] = d;
      }
    }
    for (const int d : peer.model.dof_index) {
      const auto dof = static_cast<std::size_t>(d);
      if (d >= 0 &&
          std::find(peer.base_dofs.begin(), peer.base_dofs.end(), dof) == peer.base_dofs.end() &&
          std::find(peer.required.begin(), peer.required.end(), dof) == peer.required.end()) {
        peer.required.push_back(dof);
      }
    }
    peer.joints = std::make_unique<PeerJoints>(joint_names.size());
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
        [raw](const sensor_msgs::msg::JointState::SharedPtr msg) { on_joints(*raw, *msg); });
    rclcpp::ExecutorOptions eopts;
    eopts.context = peer.context;
    peer.executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>(eopts);
    peer.executor->add_node(peer.node);
    peer.spin = std::thread([exec = peer.executor] { exec->spin(); });
  }

  static void on_joints(Peer& peer, const sensor_msgs::msg::JointState& msg) {
    const std::int64_t stamp_ns = (msg.header.stamp.sec != 0 || msg.header.stamp.nanosec != 0)
                                      ? rclcpp::Time(msg.header.stamp, RCL_ROS_TIME).nanoseconds()
                                      : peer.node->now().nanoseconds();
    std::vector<int> dof(msg.name.size(), -1);
    for (std::size_t i = 0; i < msg.name.size(); ++i) {
      const auto it = peer.dof_of.find(msg.name[i]);
      if (it != peer.dof_of.end()) {
        dof[i] = static_cast<int>(it->second);
      }
    }
    const std::vector<double> none;
    std::lock_guard<std::mutex> lock(peer.mutex);
    peer.joints->update(dof, msg.position,
                        msg.velocity.size() == msg.name.size() ? msg.velocity : none, stamp_ns);
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

  // `world_from_base` now and its planar motion over the last `kTwistWindowS`. A
  // static-only chain (a fixed arm) is stamped 0: it never ages and does not move.
  enum class Pose { kOk, kMissing, kStale, kNoHistory };
  Pose base_pose(tf2_ros::Buffer& buffer, const std::string& base, const rclcpp::Time& now,
                 tf2::Transform& world_from_base, PlanarTwist& twist, rclcpp::Time& oldest) {
    rclcpp::Time stamp;
    if (!lookup(buffer, world_frame_, base, rclcpp::Time(0, 0, RCL_ROS_TIME), world_from_base,
                stamp)) {
      return Pose::kMissing;
    }
    twist = {};
    if (stamp.nanoseconds() == 0) {
      return Pose::kOk;
    }
    if (!within_age((now - stamp).seconds(), max_age_s_)) {
      return Pose::kStale;
    }
    oldest = std::min(oldest, stamp);
    tf2::Transform before;
    rclcpp::Time before_stamp;
    if (!lookup(buffer, world_frame_, base, stamp - rclcpp::Duration::from_seconds(kTwistWindowS),
                before, before_stamp)) {
      return Pose::kNoHistory;
    }
    twist = planar_twist(before, world_from_base, kTwistWindowS);
    return Pose::kOk;
  }

  void refuse(const char* why, const std::string& who) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "peer_voxels: %s (%s): publishing nothing, the kernel refuses", why,
                         who.c_str());
  }

  static const char* pose_problem(Pose p) {
    switch (p) {
    case Pose::kMissing:
      return "no base pose in world";
    case Pose::kStale:
      return "base pose is stale or from the future";
    case Pose::kNoHistory:
      return "no base motion history";
    default:
      return "";
    }
  }

  void publish() {
    if (peers_.empty()) {
      return;
    }
    const rclcpp::Time now = this->now();
    rclcpp::Time oldest = now;
    tf2::Transform world_from_own;
    PlanarTwist own_twist;
    const Pose own = base_pose(*tf_buffer_, base_frame_, now, world_from_own, own_twist, oldest);
    if (own != Pose::kOk) {
      return refuse(pose_problem(own), base_frame_);
    }
    std::fill(grid_.occupancy.begin(), grid_.occupancy.end(), std::uint8_t{0});
    for (auto& peer : peers_) {
      std::vector<double> q;
      std::vector<double> dq;
      std::int64_t js_oldest = 0;
      std::int64_t js_newest = 0;
      {
        std::lock_guard<std::mutex> lock(peer->mutex);
        if (!peer->joints->complete(peer->required)) {
          return refuse("not every joint measured yet", peer->name);
        }
        q = peer->joints->q();
        dq = peer->joints->dq();
        js_oldest = peer->joints->oldest_ns(peer->required);
        js_newest = peer->joints->newest_ns(peer->required);
      }
      const std::int64_t now_ns = now.nanoseconds();
      if (!within_age(static_cast<double>(now_ns - js_oldest) * 1e-9, max_age_s_) ||
          !within_age(static_cast<double>(now_ns - js_newest) * 1e-9, max_age_s_)) {
        return refuse("joint states are stale or from the future", peer->name);
      }
      oldest = std::min(oldest, rclcpp::Time(js_oldest, RCL_ROS_TIME));
      tf2::Transform world_from_peer;
      PlanarTwist peer_twist;
      const Pose pose =
          base_pose(*peer->tf_buffer, peer->base_frame, now, world_from_peer, peer_twist, oldest);
      if (pose != Pose::kOk) {
        return refuse(pose_problem(pose), peer->name);
      }
      if (!peer_swept_primitives(peer->model, q, dq, peer->base_dofs, world_from_own, own_twist,
                                 world_from_peer, peer_twist, horizon_s_, prims_)) {
        return refuse("moving too fast to sweep", peer->name);
      }
      if (!rasterize_primitives(prims_, pad_m_, grid_)) {
        return refuse("a non-finite pose", peer->name);
      }
    }
    openral_msgs::msg::OccupancyVoxels msg;
    msg.header.stamp = now;
    msg.header.frame_id = base_frame_;
    msg.source_stamp = oldest;
    msg.origin.x = grid_.origin.x();
    msg.origin.y = grid_.origin.y();
    msg.origin.z = grid_.origin.z();
    msg.orientation.w = 1.0;
    msg.resolution = grid_.resolution;
    msg.size_x = msg.size_y = msg.size_z = static_cast<std::uint32_t>(grid_.n);
    msg.occupancy = grid_.occupancy;
    pub_->publish(msg);
  }

  /// How far back a base pose is read to measure its motion, s.
  static constexpr double kTwistWindowS = 0.1;

  std::string base_frame_;
  std::string world_frame_;
  tf2::Vector3 centre_;
  double radius_{0.0};
  double resolution_{0.0};
  double horizon_s_{0.0};
  double max_age_s_{0.0};
  double pad_m_{0.0};
  PeerGrid grid_;                        ///< reused every publish
  std::vector<PayloadPrimitive> prims_;  ///< reused every publish
  std::vector<std::unique_ptr<Peer>> peers_;
  rclcpp::Publisher<openral_msgs::msg::OccupancyVoxels>::SharedPtr pub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace openral_octomap_bridge

// SPDX-License-Identifier: Apache-2.0
// OctoMap → OccupancyVoxels bridge node.
//
// Subscribes an octomap_msgs/Octomap (the 3-D world map, typically in `map` /
// `odom`), deserializes it with octomap_msgs::msgToMap, looks up the transform
// from the robot base frame into the octree frame via tf2, rasterizes a bounded
// local volume around the robot into a dense base-frame occupancy grid, and
// publishes it on /openral/world_voxels for the C++ safety kernel's
// allocation-free capsule-vs-voxel check.
//
// This keeps the octomap dependency entirely out of the real-time safety kernel
// ("perception proposes, the kernel disposes").
//
// It is also where a grasped payload leaves world occupancy: the attachment set
// on /openral/world_state_fast — the same message the kernel ingests its
// attached geometry from — names the volumes whose cells stop being the world
// and become the robot's own payload, and `clear_attached_payload_cells`
// removes them from every published grid.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/time.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <openral_msgs/msg/world_state_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "openral_octomap_bridge/octree_to_grid.hpp"
#include "openral_octomap_bridge/payload_clearing.hpp"

namespace openral_octomap_bridge {

class OctomapVoxelBridge : public rclcpp::Node {
public:
  OctomapVoxelBridge() : rclcpp::Node("openral_octomap_voxel_bridge") {
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
    resolution_ = this->declare_parameter<double>("resolution", 0.05);
    box_size_[0] = this->declare_parameter<double>("box_size_x", 2.0);
    box_size_[1] = this->declare_parameter<double>("box_size_y", 2.0);
    box_size_[2] = this->declare_parameter<double>("box_size_z", 2.0);
    box_center_[0] = this->declare_parameter<double>("box_center_x", 0.0);
    box_center_[1] = this->declare_parameter<double>("box_center_y", 0.0);
    box_center_[2] = this->declare_parameter<double>("box_center_z", 0.5);
    const auto octomap_topic =
        this->declare_parameter<std::string>("octomap_topic", "/octomap_binary");
    const auto output_topic =
        this->declare_parameter<std::string>("output_topic", "/openral/world_voxels");
    const double rate_hz = this->declare_parameter<double>("publish_rate_hz", 10.0);

    // Attached-payload clearing. On by default: `AttachedCollisionObject`
    // requires the object to be absent from world occupancy while attached, and
    // with no attachment producer on the graph this costs one empty check.
    attached_clear_enabled_ = this->declare_parameter<bool>("attached_clear_enabled", true);
    // Extra reach beyond the cell-cube circumradius. 0 clears exactly the cells
    // the payload's own volume can explain; anything more removes cells it
    // cannot, and is protection given up for pose uncertainty.
    attached_clear_padding_m_ = this->declare_parameter<double>("attached_clear_padding_m", 0.0);
    attached_state_timeout_s_ = this->declare_parameter<double>("attached_state_timeout_s", 0.5);
    const auto world_state_topic =
        this->declare_parameter<std::string>("world_state_topic", "/openral/world_state_fast");

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
        octomap_topic, rclcpp::QoS(1).reliable(),
        std::bind(&OctomapVoxelBridge::on_octomap, this, std::placeholders::_1));
    voxel_pub_ = this->create_publisher<openral_msgs::msg::OccupancyVoxels>(
        output_topic, rclcpp::QoS(1).reliable());
    if (attached_clear_enabled_) {
      // Matches the kernel's own subscription to this topic (RELIABLE,
      // VOLATILE, KEEP_LAST=1 at 30 Hz), so bridge and kernel act on the same
      // attachment set.
      world_state_sub_ = this->create_subscription<openral_msgs::msg::WorldStateStamped>(
          world_state_topic, rclcpp::QoS(1).reliable(),
          [this](openral_msgs::msg::WorldStateStamped::SharedPtr msg) {
            world_state_ = std::move(msg);
          });
    }
    timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / std::max(rate_hz, 1.0)),
                                     std::bind(&OctomapVoxelBridge::on_timer, this));

    RCLCPP_INFO(this->get_logger(),
                "octomap→voxel bridge: %s → %s, base=%s, box=[%g %g %g]@[%g %g "
                "%g], res=%g",
                octomap_topic.c_str(), output_topic.c_str(), base_frame_.c_str(), box_size_[0],
                box_size_[1], box_size_[2], box_center_[0], box_center_[1], box_center_[2],
                resolution_);
  }

private:
  void on_octomap(const octomap_msgs::msg::Octomap::SharedPtr msg) {
    // octomap_msgs::msgToMap handles both binary and full encodings and returns
    // a heap-allocated AbstractOcTree the caller owns.
    octomap::AbstractOcTree* abstract = octomap_msgs::msgToMap(*msg);
    if (abstract == nullptr) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "failed to deserialize octomap message");
      return;
    }
    auto* octree = dynamic_cast<octomap::OcTree*>(abstract);
    if (octree == nullptr) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "octomap is not an OcTree (id=%s); only OcTree is supported",
                           msg->id.c_str());
      delete abstract;
      return;
    }
    octree_.reset(octree);  // takes ownership
    octomap_frame_ = msg->header.frame_id;
  }

  void on_timer() {
    if (octree_ == nullptr || octomap_frame_.empty()) {
      return;
    }
    geometry_msgs::msg::TransformStamped tf_msg;
    try {
      tf_msg = tf_buffer_->lookupTransform(octomap_frame_, base_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "TF %s <- %s unavailable: %s", octomap_frame_.c_str(),
                           base_frame_.c_str(), ex.what());
      return;
    }
    tf2::Transform base_to_octomap;
    tf2::fromMsg(tf_msg.transform, base_to_octomap);

    GridSpec spec;
    spec.resolution = resolution_;
    spec.sx = static_cast<std::uint32_t>(std::ceil(box_size_[0] / resolution_));
    spec.sy = static_cast<std::uint32_t>(std::ceil(box_size_[1] / resolution_));
    spec.sz = static_cast<std::uint32_t>(std::ceil(box_size_[2] / resolution_));
    spec.box_min[0] = box_center_[0] - 0.5 * box_size_[0];
    spec.box_min[1] = box_center_[1] - 0.5 * box_size_[1];
    spec.box_min[2] = box_center_[2] - 0.5 * box_size_[2];

    auto grid = rasterize_octree_to_grid(*octree_, base_to_octomap, spec, base_frame_);
    clear_attached_payload(grid);
    grid.header.stamp = this->now();
    voxel_pub_->publish(grid);
  }

  /// Remove the grasped payload's own cells from the grid about to be
  /// published, in the payload's live pose.
  ///
  /// Every way of not knowing where the payload is — no attachment message, a
  /// stale one, a missing attach-link TF, a primitive the kernel itself would
  /// refuse — clears **nothing** and leaves the map exactly as the octree
  /// described it. That is the conservative failure: the map keeps occupancy it
  /// should not have (which can only stop the robot early), never loses
  /// occupancy it should have.
  void clear_attached_payload(openral_msgs::msg::OccupancyVoxels& grid) {
    if (!attached_clear_enabled_) {
      return;
    }
    const auto state = world_state_;
    if (state == nullptr || state->attached_objects.empty()) {
      return;
    }
    const rclcpp::Time stamp(state->header.stamp);
    const double age_s = (this->now() - stamp).seconds();
    if (age_s > attached_state_timeout_s_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "attachment state is %.3f s old (> %.3f s): clearing nothing", age_s,
                           attached_state_timeout_s_);
      return;
    }

    std::vector<PayloadPrimitive> primitives;
    for (const auto& object : state->attached_objects) {
      geometry_msgs::msg::TransformStamped tf_msg;
      try {
        tf_msg = tf_buffer_->lookupTransform(base_frame_, object.attach_link, tf2::TimePointZero);
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "TF %s <- %s unavailable (%s): payload %s stays in the map",
                             base_frame_.c_str(), object.attach_link.c_str(), ex.what(),
                             object.object_id.c_str());
        return;
      }
      tf2::Transform base_from_link;
      tf2::fromMsg(tf_msg.transform, base_from_link);
      if (!place_attached_object(object, base_from_link, primitives)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "payload %s carries geometry this bridge cannot place: clearing "
                             "nothing",
                             object.object_id.c_str());
        return;
      }
    }

    const std::size_t cleared =
        clear_attached_payload_cells(grid, primitives, attached_clear_padding_m_);
    if (cleared > 0) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "attached payload: cleared %zu cell(s) of %zu object(s) from world "
                           "occupancy",
                           cleared, state->attached_objects.size());
    }
  }

  std::string base_frame_;
  std::string octomap_frame_;
  double resolution_{0.05};
  double box_size_[3]{2.0, 2.0, 2.0};
  double box_center_[3]{0.0, 0.0, 0.5};
  bool attached_clear_enabled_{true};
  double attached_clear_padding_m_{0.0};
  double attached_state_timeout_s_{0.5};

  std::unique_ptr<octomap::OcTree> octree_;
  openral_msgs::msg::WorldStateStamped::SharedPtr world_state_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
  rclcpp::Subscription<openral_msgs::msg::WorldStateStamped>::SharedPtr world_state_sub_;
  rclcpp::Publisher<openral_msgs::msg::OccupancyVoxels>::SharedPtr voxel_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace openral_octomap_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<openral_octomap_bridge::OctomapVoxelBridge>());
  rclcpp::shutdown();
  return 0;
}

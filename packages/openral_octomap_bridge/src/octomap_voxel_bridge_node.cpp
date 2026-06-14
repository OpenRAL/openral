// SPDX-License-Identifier: Apache-2.0
// ADR-0030 — OctoMap → OccupancyVoxels bridge node.
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/time.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <rclcpp/rclcpp.hpp>

#include "openral_octomap_bridge/octree_to_grid.hpp"

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

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
        octomap_topic, rclcpp::QoS(1).reliable(),
        std::bind(&OctomapVoxelBridge::on_octomap, this, std::placeholders::_1));
    voxel_pub_ = this->create_publisher<openral_msgs::msg::OccupancyVoxels>(
        output_topic, rclcpp::QoS(1).reliable());
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
    grid.header.stamp = this->now();
    voxel_pub_->publish(grid);
  }

  std::string base_frame_;
  std::string octomap_frame_;
  double resolution_{0.05};
  double box_size_[3]{2.0, 2.0, 2.0};
  double box_center_[3]{0.0, 0.0, 0.5};

  std::unique_ptr<octomap::OcTree> octree_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
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

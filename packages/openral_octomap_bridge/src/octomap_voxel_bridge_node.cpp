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
// removes them from every published grid. The same message carries the
// support-contact attestation, and the clearing is partitioned against it: the
// cells inside the attested support patch are the counter the payload rests on,
// so they stay in the map for the kernel's witness (and for Nav2 / SLAM) while
// everything else around the payload clears.
//
// The message also carries `attachment_revision`, and that is what opens the
// attach-transition WINDOW: while a payload is still within one sweep padding
// of where its revision first appeared, it sweeps with `attach_sweep_padding_m`
// of extra reach, for the stale pre-attach silhouette that primitive-fit error
// leaves just outside the steady-state reach. Once it has moved further than
// that, the window latches shut and the reach is the steady one again.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
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
    // The grid's resolution is the OCTREE's — the two lattices are one now, so
    // there is nothing left to choose. This parameter survives only as the
    // default for `attach_sweep_padding_m` below, which has to be declared
    // before any octree has arrived; `on_timer` warns if the map disagrees.
    resolution_ = this->declare_parameter<double>("resolution", 0.05);
    // The volume covered: a BALL, because the grid's axes are the map's and so
    // turn relative to `base_frame`. See GridSpec — a base-frame box would need
    // its rotated bounding box covered, costing up to 2x the cells at 45
    // degrees and making the cell count depend on the robot's heading.
    coverage_center_[0] = this->declare_parameter<double>("coverage_center_x", 0.0);
    coverage_center_[1] = this->declare_parameter<double>("coverage_center_y", 0.0);
    coverage_center_[2] = this->declare_parameter<double>("coverage_center_z", 0.5);
    // No default that publishes. The radius must cover wherever the kernel's
    // checked geometry can reach, which is a property of the ROBOT — this node
    // cannot derive it, and a guessed one is a grid that is silently blind
    // where the arm actually goes. Unset means publish nothing.
    coverage_radius_m_ = this->declare_parameter<double>("coverage_radius_m", 0.0);
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
    // Extra reach on the ONE grid that first sees an object revision — the
    // attach transition. One voxel by default, because the residue it exists
    // for is a quantization/primitive-fit artefact of exactly that scale (the
    // 2026-08-14 round-5 cell sat 0.48 mm past the circumradius). Declared
    // against `resolution` so the default is one voxel of whatever lattice this
    // node is actually publishing, and it shows up as a real number in
    // `ros2 param get`.
    attach_sweep_padding_m_ =
        this->declare_parameter<double>("attach_sweep_padding_m", resolution_);
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

    if (!(coverage_radius_m_ > 0.0)) {
      RCLCPP_ERROR(this->get_logger(),
                   "coverage_radius_m is unset (%g): publishing NOTHING. Set it to cover the "
                   "robot's kernel-checked geometry — a smaller grid is one the safety kernel "
                   "is blind outside of.",
                   coverage_radius_m_);
    }
    RCLCPP_INFO(this->get_logger(),
                "octomap→voxel bridge: %s → %s, base=%s, covering r=%g m @[%g %g %g], "
                "lattice=the octree's",
                octomap_topic.c_str(), output_topic.c_str(), base_frame_.c_str(),
                coverage_radius_m_, coverage_center_[0], coverage_center_[1], coverage_center_[2]);
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

    if (!(coverage_radius_m_ > 0.0)) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                            "coverage_radius_m is unset: publishing nothing");
      return;
    }
    if (std::fabs(octree_->getResolution() - resolution_) > 1e-9) {
      // Not fatal — the grid takes the octree's lattice regardless — but the
      // `resolution` parameter still sets the attach-sweep padding, so a
      // disagreement means that padding is not one voxel of the live map.
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "octree resolution %g != `resolution` parameter %g; the grid uses the "
                           "octree's, but attach_sweep_padding_m was defaulted from the parameter",
                           octree_->getResolution(), resolution_);
    }

    GridSpec spec;
    spec.center[0] = coverage_center_[0];
    spec.center[1] = coverage_center_[1];
    spec.center[2] = coverage_center_[2];
    spec.radius = coverage_radius_m_;

    auto grid = rasterize_octree_to_grid(*octree_, base_to_octomap, spec, base_frame_);
    if (grid.occupancy.empty()) {
      // The lattice could not be placed (an unusable radius, or a spec larger
      // than the rasterizer will allocate). Publishing the empty grid would be
      // worse than publishing nothing: a zero-cell grid reads downstream as a
      // world with no obstacles in it, while silence is what the kernel's
      // staleness watchdog exists to catch.
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                            "could not place a grid lattice for radius %g m at the octree's %g m "
                            "resolution: publishing nothing",
                            coverage_radius_m_, octree_->getResolution());
      return;
    }
    clear_attached_payload(grid, base_to_octomap);
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
  ///
  /// Objects are split into two clearing passes by the ONE piece of state this
  /// node keeps (`AttachSweepLedger`): a payload whose attach-transition window
  /// is still open sweeps at the padded reach, one whose window has closed at
  /// the unchanged steady-state reach. Both passes see the full patch list, so
  /// a padded sweep withholds the attested support patch exactly as an ordinary
  /// one does.
  ///
  /// `base_to_octomap` is `lookupTransform(octomap_frame, base_frame)`, already
  /// looked up for the rasterization. The window measures the payload's
  /// displacement in the OctoMap's frame, not the base frame, because the stale
  /// silhouette it is chasing is world-fixed occupancy: a payload motionless in
  /// `base_link` on a driving mobile base has left its silhouette behind, and
  /// the window must close.
  void clear_attached_payload(openral_msgs::msg::OccupancyVoxels& grid,
                              const tf2::Transform& base_to_octomap) {
    if (!attached_clear_enabled_) {
      return;
    }
    const auto state = world_state_;
    if (state == nullptr) {
      return;
    }
    if (state->attached_objects.empty()) {
      // Nothing attached: nothing to sweep and no window to keep, so the next
      // grasp — of this object or any other — opens a fresh one.
      attach_sweep_ledger_.sweep({}, attach_sweep_padding_m_);
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

    std::vector<SupportPatch> patches;
    // Staged per object, because the window is decided per object.
    std::vector<std::vector<PayloadPrimitive>> staged;
    std::vector<AttachSweepObservation> present;
    staged.reserve(state->attached_objects.size());
    present.reserve(state->attached_objects.size());
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
      std::vector<PayloadPrimitive> object_primitives;
      if (!place_attached_object(object, base_from_link, object_primitives, patches)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "payload %s carries geometry or a support attestation this bridge "
                             "cannot place: clearing nothing",
                             object.object_id.c_str());
        return;
      }
      // The window's reference point is the first primitive's origin lifted
      // into the OctoMap frame — a material point of the payload, which is all
      // "has this payload moved away from its silhouette?" needs. A successful
      // placement always yields at least one primitive.
      present.push_back(AttachSweepObservation{
          AttachedObjectRevision{object.object_id, state->attachment_revision},
          base_to_octomap * object_primitives.front().pose.getOrigin()});
      staged.push_back(std::move(object_primitives));
    }

    // Answering and recording in one call is what keeps the window's state and
    // the reach it licenses from disagreeing. Every early return above skips it,
    // so a frame that clears nothing leaves the padded sweep owed.
    const std::vector<std::uint8_t> window_open =
        attach_sweep_ledger_.sweep(present, attach_sweep_padding_m_);
    std::vector<PayloadPrimitive> steady_primitives;
    std::vector<PayloadPrimitive> attach_sweep_primitives;
    std::size_t attach_sweep_objects = 0;
    for (std::size_t i = 0; i < staged.size(); ++i) {
      auto& group = window_open[i] != 0 ? attach_sweep_primitives : steady_primitives;
      group.insert(group.end(), staged[i].begin(), staged[i].end());
      attach_sweep_objects += window_open[i] != 0 ? 1U : 0U;
    }

    const double attach_padding =
        attach_transition_padding(attached_clear_padding_m_, attach_sweep_padding_m_);
    const std::size_t cleared =
        clear_attached_payload_cells(grid, steady_primitives, patches, attached_clear_padding_m_) +
        clear_attached_payload_cells(grid, attach_sweep_primitives, patches, attach_padding);
    if (cleared > 0) {
      // The attested-patch count is logged unconditionally beside the cleared
      // count: the partition is the difference between the two numbers, and a
      // support surface silently vanishing from the map is exactly what this
      // line exists to make visible (CLAUDE.md §1.4). The open-window count is
      // beside them for the same reason — a window that never closes is a
      // widened reach that has quietly become the steady state, and a field
      // trace has to be able to show when it shut.
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "attached payload: cleared %zu cell(s) of %zu object(s) from world "
                           "occupancy, withholding %zu attested support patch(es); %zu object(s) "
                           "inside the attach window (+%.4f m, revision %lu)",
                           cleared, state->attached_objects.size(), patches.size(),
                           attach_sweep_objects, attach_padding,
                           static_cast<unsigned long>(state->attachment_revision));
    }
  }

  std::string base_frame_;
  std::string octomap_frame_;
  double resolution_{0.05};
  double coverage_center_[3]{0.0, 0.0, 0.5};
  double coverage_radius_m_{0.0};
  bool attached_clear_enabled_{true};
  double attached_clear_padding_m_{0.0};
  double attach_sweep_padding_m_{0.05};
  double attached_state_timeout_s_{0.5};
  AttachSweepLedger attach_sweep_ledger_;

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

// SPDX-License-Identifier: Apache-2.0
// OctoMap → OccupancyVoxels bridge node.
//
// The node class lives in this private header (not `include/`) so the gtests
// can run the real node in-process; `octomap_voxel_bridge_node.cpp` is its
// `main`.
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
//
// The timer re-rasterizes the LAST octree, so the grid follows the robot
// through TF and the payload clearing follows its live pose between octrees.
// That republish is bounded by `max_octree_age_s`: `octomap_server` publishes
// only when it inserts a cloud, so a dead camera is a silent octree, and a
// frozen map republished on time would never let the kernel's
// `world_voxel_deadline_ms` fire. Past the bound this node publishes nothing —
// not even a payload-cleared grid — and the kernel fails closed with
// `DROP_VOXEL_UNAVAILABLE` (hazard log Entry 033).

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/srv/bounding_box_query.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/time.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <openral_msgs/msg/world_state_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "openral_octomap_bridge/identity_clearing.hpp"
#include "openral_octomap_bridge/octree_freshness.hpp"
#include "openral_octomap_bridge/octree_to_grid.hpp"
#include "openral_octomap_bridge/payload_clearing.hpp"

namespace openral_octomap_bridge {

class OctomapVoxelBridge : public rclcpp::Node {
public:
  explicit OctomapVoxelBridge(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : rclcpp::Node("openral_octomap_voxel_bridge", options) {
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
    // How long the last octree may keep being republished after it arrived.
    // Must exceed octomap's normal inter-publish gap with margin (0.25-0.31 s on
    // the Thor ZED path at 3.2-4.0 Hz, ~0.45 s at 2.2 Hz; ~0.1-0.33 s in sim) and
    // not exceed the kernel's `world_voxel_deadline_ms`, which is what turns the
    // resulting silence into a fail-closed drop. `deploy_e2e.launch.py` sets it
    // equal to that deadline (1.0 s); worst case from the last cloud to the drop
    // is bound + deadline, 2.0 s (hazard log Entry 033). Receipt time, on this
    // node's clock: the clock the kernel times voxel freshness on, immune to a
    // sensor stamping on another clock domain.
    max_octree_age_s_ = this->declare_parameter<double>("max_octree_age_s", 1.0);

    // Attached-payload clearing. On by default: `AttachedCollisionObject`
    // requires the object to be absent from world occupancy while attached, and
    // with no attachment producer on the graph this costs one empty check.
    // Declared static geometry (docs/reference/manipulation-workspace-contact.md
    // §3.1): base-frame surface points of the scene's fixtures, marked occupied in
    // every published grid whatever the camera has seen. Only adds occupancy.
    // Empty = off. A file that does not load is fatal: a deploy that declared its
    // walls must not run without them.
    static_points_file_ = this->declare_parameter<std::string>("static_points_file", "");
    if (!static_points_file_.empty()) {
      std::string error;
      if (!load_points_file(static_points_file_, static_points_, error)) {
        throw std::runtime_error("static_points_file: " + error);
      }
      RCLCPP_WARN(this->get_logger(),
                  "declared static geometry: %zu base-frame surface points from %s are marked "
                  "occupied in every published grid",
                  static_points_.size(), static_points_file_.c_str());
    }
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
    // The carried payload's own posed volume, cleared from the OCTREE (`SweptBox`): the
    // grid clearing above follows only the payload's current pose, and what the carried
    // object left along its path stays in the octree. Empty service = off. User-approved
    // 2026-10-16; Safety-WG review and a hazard-log row owed (the payload's volume is
    // cleared from the world map). Only poses the payload is attached in are cleared:
    // there is no pre-attach backfill, since the hand's path before the attach is where
    // the object never was.
    const auto sweep_service =
        this->declare_parameter<std::string>("payload_sweep_clear_service", "");
    sweep_period_s_ = this->declare_parameter<double>("payload_sweep_clear_period_s", 0.2);
    if (!sweep_service.empty()) {
      sweep_client_ = this->create_client<octomap_msgs::srv::BoundingBoxQuery>(sweep_service);
      RCLCPP_WARN(this->get_logger(),
                  "payload sweep clearing ON: the carried payload's posed primitive boxes "
                  "(every %.2f s, attached poses only) are cleared from the octree via %s",
                  sweep_period_s_, sweep_service.c_str());
    }
    const auto world_state_topic =
        this->declare_parameter<std::string>("world_state_topic", "/openral/world_state_fast");
    // Manifest attach link -> the frame the live TF tree names that body
    // ("link=frame"; the scene's `vision_attachment.tf_frames`, the same strings
    // the HAL gets). Empty: every attach link is looked up by its own name.
    const auto tf_frame_entries = this->declare_parameter<std::vector<std::string>>(
        "attach_link_tf_frames", std::vector<std::string>{});
    std::string tf_frames_error;
    if (!parse_attach_link_tf_frames(tf_frame_entries, attach_link_tf_frames_, tf_frames_error)) {
      // Identity for every link: a renamed hand then fails its lookup and its
      // payload stays in the map — fail closed, never cleared through a misread.
      RCLCPP_ERROR(this->get_logger(),
                   "attach_link_tf_frames: %s — ignoring the whole mapping; a held payload whose "
                   "attach link is not itself a TF frame stays in the map",
                   tf_frames_error.c_str());
    }

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
    if (!valid_max_octree_age(max_octree_age_s_)) {
      RCLCPP_ERROR(this->get_logger(),
                   "max_octree_age_s must be in (0, %g] (got %g): publishing NOTHING — a "
                   "bound that cannot be applied, or one past the kernel's deadline cap, is a "
                   "map of unknown age.",
                   kMaxOctreeAgeS, max_octree_age_s_);
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
    octree_received_ = this->now();
    // `octomap_server` stamps the octree with the capture stamp of the cloud it
    // inserted; carried into every grid built from it as `source_stamp`, so the
    // kernel can budget the age of the WORLD the grid describes, which the
    // receipt-based liveness bound above cannot see.
    octree_stamp_ = rclcpp::Time(msg->header.stamp, this->get_clock()->get_clock_type());
  }

  void on_timer() {
    if (octree_ == nullptr || octomap_frame_.empty()) {
      return;
    }
    // Before anything else, so a stale octree reaches no grid — a
    // payload-cleared one included.
    const double octree_age_s = (this->now() - octree_received_).seconds();
    if (!octree_is_fresh(octree_age_s, max_octree_age_s_)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "last octree is %.3f s old (max_octree_age_s %g): publishing nothing "
                           "until the next one, so the kernel's world_voxel_deadline_ms fails "
                           "closed (DROP_VOXEL_UNAVAILABLE) instead of trusting a frozen map",
                           octree_age_s, max_octree_age_s_);
      octree_stale_ = true;
      return;
    }
    if (octree_stale_) {
      RCLCPP_INFO(this->get_logger(), "fresh octree (%.3f s old): publishing again", octree_age_s);
      octree_stale_ = false;
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
    if (!static_points_.empty()) {
      // Before the payload clearing, so a declared surface under a carried
      // payload is partitioned against its support exactly like a seen one.
      const std::size_t added = mark_base_points(grid, static_points_);
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "declared static geometry added %zu cells the camera map lacked", added);
    }
    clear_attached_payload(grid, base_to_octomap);
    clear_by_identity(grid, base_to_octomap);
    clear_swept_volume(base_to_octomap);
    const rclcpp::Time produced = this->now();
    grid.header.stamp = produced;
    grid.source_stamp = octree_stamp_;
    voxel_pub_->publish(grid);
    // The one number that says how far behind the world this grid is: capture
    // of the newest cloud in it to now. Throttled; the kernel carries the same
    // age on every check span as `safety.world_voxel_data_age_ms`.
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "world_voxels data age %.0f ms at publish (octree received %.0f ms ago)",
                         (produced - octree_stamp_).seconds() * 1e3, octree_age_s * 1e3);
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
      support_patch_anchors_.anchor({});
      swept_seen_.clear();
      sweep_supports_ = SweepSupports{};
      sweep_support_keys_.clear();
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
    // Each attestation's key and its patch in the OctoMap frame, for the anchors.
    std::vector<SupportPatchObservation> attested;
    // Parallel to `staged`: the objects this (pre-#418) path clears.
    std::vector<const openral_msgs::msg::AttachedCollisionObject*> legacy;
    staged.reserve(state->attached_objects.size());
    present.reserve(state->attached_objects.size());
    for (const auto& object : state->attached_objects) {
      // A tracker binding (#418) is cleared by identity (`clear_by_identity`) or not
      // at all: a pinch fallback or an ambiguous binding stays an obstacle.
      if (object.evidence_kind == kTrackedObjectEvidence ||
          object.evidence_kind == kPinchFallbackEvidence) {
        continue;
      }
      geometry_msgs::msg::TransformStamped tf_msg;
      const std::string& link_frame = tf_frame_for(attach_link_tf_frames_, object.attach_link);
      try {
        tf_msg = tf_buffer_->lookupTransform(base_frame_, link_frame, tf2::TimePointZero);
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "TF %s <- %s unavailable (%s): payload %s stays in the map",
                             base_frame_.c_str(), link_frame.c_str(), ex.what(),
                             object.object_id.c_str());
        if (attach_link_tf_frames_.count(object.attach_link) == 0 &&
            unmapped_links_warned_.insert(object.attach_link).second) {
          RCLCPP_WARN(this->get_logger(),
                      "attach link %s is not a TF frame and attach_link_tf_frames does not map "
                      "it: if the cell's TF tree names this body differently, set "
                      "attach_link_tf_frames ['%s=<tf frame>'] (the scene's "
                      "vision_attachment.tf_frames)",
                      object.attach_link.c_str(), object.attach_link.c_str());
        }
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
      if (object.support_contact_valid) {
        // `place_attached_object` appended this object's patch last.
        SupportPatch world = patches.back();
        world.point = base_to_octomap * world.point;
        world.normal = base_to_octomap.getBasis() * world.normal;
        attested.push_back(SupportPatchObservation{
            SupportWitnessKey{object.object_id, object.support_contact.support_id,
                              object.support_contact.stamp_ns},
            world});
      }
      present.push_back(AttachSweepObservation{
          AttachedObjectRevision{object.object_id, state->attachment_revision},
          base_to_octomap * object_primitives.front().pose.getOrigin()});
      staged.push_back(std::move(object_primitives));
      legacy.push_back(&object);
    }

    if (sweep_client_) {
      accumulate_swept_volume(*state, legacy, staged, base_to_octomap, attested);
    }

    // Answering and recording in one call is what keeps the window's state and
    // the reach it licenses from disagreeing. Every early return above skips it,
    // so a frame that clears nothing leaves the padded sweep owed.
    const std::vector<std::uint8_t> window_open =
        attach_sweep_ledger_.sweep(present, attach_sweep_padding_m_);
    // The band stays where the support was attested (`SupportPatchAnchors`):
    // each patch withholds only what its first-grid plane also claims.
    const std::vector<SupportPatch> anchors = support_patch_anchors_.anchor(attested);
    const tf2::Transform octomap_to_base = base_to_octomap.inverse();
    for (std::size_t i = 0; i < anchors.size(); ++i) {
      SupportPatch& patch = patches[i];
      patch.has_anchor = true;
      patch.anchor_point = octomap_to_base * anchors[i].point;
      patch.anchor_normal = octomap_to_base.getBasis() * anchors[i].normal;
    }
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

  /// Record this tick's posed primitive boxes, the supports attested during the carry,
  /// and, at an object's first appearance (a new attachment revision), a REGION
  /// payload's lowest point. No pre-attach TF backfill: posing the payload rigidly in
  /// the hand along the approach clears where the object never was.
  void accumulate_swept_volume(
      const openral_msgs::msg::WorldStateStamped& state,
      const std::vector<const openral_msgs::msg::AttachedCollisionObject*>& objects,
      const std::vector<std::vector<PayloadPrimitive>>& staged,
      const tf2::Transform& base_to_octomap, const std::vector<SupportPatchObservation>& attested) {
    for (const auto& prims : staged) {
      const auto boxes = payload_primitive_boxes(prims, base_to_octomap);
      swept_boxes_.insert(swept_boxes_.end(), boxes.begin(), boxes.end());
    }
    // Latch every support attested during this carry, as first attested: a vision
    // pick's support attestation lasts only "until the payload moves", and the carry's
    // later boxes must still stop above that counter.
    for (const auto& obs : attested) {
      if (sweep_support_keys_
              .insert(obs.key.object_id + "|" + obs.key.support_id + "|" +
                      std::to_string(obs.key.stamp_ns))
              .second) {
        sweep_supports_.patches.push_back(obs.patch);
      }
    }
    for (std::size_t i = 0; i < objects.size(); ++i) {
      const auto& object = *objects[i];
      if (!swept_seen_.insert({object.object_id, state.attachment_revision}).second) {
        continue;
      }
      // A REGION payload's lowest point at attach floors a box no support is known under:
      // it is the producer-measured grasp-target region, stood on its measured support.
      // A segmented payload's bottom is not (deploy trial g17: its points reached into
      // the counter and depth edges, floors at z 0.020-0.027 under a 0.030 counter top),
      // and the gripper-closure jaw box is not the object: neither floors anything.
      if (object.evidence_kind == "grasp_target_region" && i < staged.size()) {
        SweptBox at_attach;
        add_payload_to_swept_box(at_attach, staged[i], base_to_octomap);
        if (at_attach.valid) {
          sweep_supports_.measured_floor_z =
              sweep_supports_.has_measured_floor
                  ? std::min(sweep_supports_.measured_floor_z, at_attach.lo.z())
                  : at_attach.lo.z();
          sweep_supports_.has_measured_floor = true;
        }
      }
    }
  }

  /// Every `payload_sweep_clear_period_s`: floor the period's posed payload boxes above
  /// every known support (`floor_pose_boxes`) and ask `octomap_server` to clear each from
  /// the octree; then start a new period.
  void clear_swept_volume(const tf2::Transform& base_to_octomap) {
    if (!sweep_client_ || swept_boxes_.empty()) {
      return;
    }
    const rclcpp::Time now = this->now();
    if (last_sweep_clear_.nanoseconds() != 0 &&
        (now - last_sweep_clear_).seconds() < sweep_period_s_) {
      return;
    }
    std::vector<SweptBox> boxes;
    boxes.swap(swept_boxes_);
    last_sweep_clear_ = now;
    // Declared static geometry is in the base frame: carried into the octomap frame here,
    // once per clear, not on every grid tick.
    sweep_supports_.static_points.clear();
    sweep_supports_.static_points.reserve(static_points_.size());
    for (const auto& p : static_points_) {
      sweep_supports_.static_points.push_back(base_to_octomap * p);
    }
    std::vector<SweptBox> clears;
    const SweepFloor floor = floor_pose_boxes(boxes, sweep_supports_, resolution_, clears);
    if (clears.empty()) {
      RCLCPP_INFO(this->get_logger(),
                  "payload sweep: NOT clearing %zu posed box(es): no support attested or "
                  "declared under them and no region payload floor (a segmented or "
                  "gripper-closure payload), or nothing left above the support",
                  boxes.size());
      return;
    }
    if (!sweep_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "payload sweep: %s not available; the payload volume stays in the "
                           "octree",
                           sweep_client_->get_service_name());
      return;
    }
    // A request the server never answers would stay pending in the client forever; drop
    // any older than kSweepRequestTimeout and say so (the volume then stays in the octree).
    const size_t pruned = sweep_client_->prune_requests_older_than(
        std::chrono::system_clock::now() - kSweepRequestTimeout);
    if (pruned > 0) {
      RCLCPP_WARN(this->get_logger(),
                  "payload sweep: %zu clear_bbox request(s) to %s unanswered after %llds; "
                  "dropped, their volume stays in the octree",
                  pruned, sweep_client_->get_service_name(),
                  static_cast<long long>(kSweepRequestTimeout.count()));
    }
    for (const auto& box : clears) {
      auto req = std::make_shared<octomap_msgs::srv::BoundingBoxQuery::Request>();
      req->min.x = box.lo.x();
      req->min.y = box.lo.y();
      req->min.z = box.lo.z();
      req->max.x = box.hi.x();
      req->max.y = box.hi.y();
      req->max.z = box.hi.z();
      sweep_client_->async_send_request(req);
      RCLCPP_INFO(this->get_logger(),
                  "payload sweep: clear_bbox [%.3f %.3f %.3f]..[%.3f %.3f %.3f] in %s (floor: %s)",
                  box.lo.x(), box.lo.y(), box.lo.z(), box.hi.x(), box.hi.y(), box.hi.z(),
                  octomap_frame_.c_str(),
                  floor == SweepFloor::kSupport ? "support" : "region payload at attach");
    }
  }

  struct IdentityCarry {
    std::vector<tf2::Transform> pending;  ///< octomap <- link, not yet cleared
    bool has_last{false};
    std::int64_t last_ns{0};
    tf2::Transform last{tf2::Transform::getIdentity()};
    double reach_m{0.0};  ///< furthest geometry point from the link origin
  };
  struct IdentityHome {
    std::uint32_t epoch{0};
    PayloadPrimitive box;  ///< last confirmed box, octomap frame
  };

  /// Clearing by identity (issue #418, `identity_clearing.hpp`). On the grid, each
  /// unambiguous tracker binding's cells at its live pose. On the octree (with
  /// `payload_sweep_clear_service` set), every pose since `t_attach` — backfilled from
  /// TF history, interpolated half a cell apart — and, once, the box a tracked object
  /// has left. Every failure (stale or unusable set, missing TF, geometry the bridge
  /// will not place) clears less, never more.
  void clear_by_identity(openral_msgs::msg::OccupancyVoxels& grid,
                         const tf2::Transform& base_to_octomap) {
    const auto state = world_state_;
    if (!attached_clear_enabled_ || state == nullptr) {
      return;
    }
    const double age_s = (this->now() - rclcpp::Time(state->header.stamp)).seconds();
    if (age_s > attached_state_timeout_s_ || !tracked_set_usable(*state, base_frame_)) {
      // Pending poses and remembered boxes dropped: less is cleared, and a box is never
      // vacated across a gap in which nobody watched it.
      identity_carries_.clear();
      identity_homes_.clear();
      return;
    }
    const auto& set = state->tracked_objects;
    const rclcpp::Time now = this->now();
    const bool octree_work = sweep_client_ != nullptr && octree_ != nullptr;
    const bool flush = octree_work && (last_identity_flush_.nanoseconds() == 0 ||
                                       (now - last_identity_flush_).seconds() >= sweep_period_s_);
    if (octree_work) {
      static_keys_.clear();
      for (const auto& p : static_points_) {
        const tf2::Vector3 w = base_to_octomap * p;
        octomap::OcTreeKey key;
        if (octree_->coordToKeyChecked(octomap::point3d(static_cast<float>(w.x()),
                                                        static_cast<float>(w.y()),
                                                        static_cast<float>(w.z())),
                                       key)) {
          static_keys_.insert(key);
        }
      }
    }
    if (flush) {
      last_identity_flush_ = now;
    }
    std::map<std::pair<std::string, std::uint32_t>, IdentityCarry> carries;
    std::map<std::string, std::vector<PayloadPrimitive>> live;  // bound, octomap frame
    std::size_t grid_cleared = 0;
    for (const auto& object : state->attached_objects) {
      const auto* track = identity_clearing_track(object, *state, base_frame_);
      if (track == nullptr) {
        continue;
      }
      const std::string& link_frame = tf_frame_for(attach_link_tf_frames_, object.attach_link);
      geometry_msgs::msg::TransformStamped tf_msg;
      try {
        tf_msg = tf_buffer_->lookupTransform(base_frame_, link_frame, tf2::TimePointZero);
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "identity clear: TF %s <- %s unavailable (%s): %s stays in the map",
                             base_frame_.c_str(), link_frame.c_str(), ex.what(),
                             object.object_id.c_str());
        continue;
      }
      tf2::Transform base_from_link;
      tf2::fromMsg(tf_msg.transform, base_from_link);
      std::vector<IdentityPiece> pieces;
      IdentityKeepOut keep;
      if (!place_identity_pieces(object, base_from_link, pieces) ||
          !identity_keep_out(set, *track, tf2::Transform::getIdentity(), false, keep)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "identity clear: %s carries geometry or a track set this bridge "
                             "cannot place: clearing nothing for it",
                             object.object_id.c_str());
        continue;
      }
      grid_cleared += clear_identity_cells(grid, pieces, keep);
      if (!octree_work) {
        continue;
      }
      const tf2::Transform octomap_from_link = base_to_octomap * base_from_link;
      for (const auto& piece : pieces) {
        PayloadPrimitive placed = piece.prim;
        placed.pose = base_to_octomap * placed.pose;
        live[object.object_id].push_back(placed);
      }
      const auto key = std::make_pair(object.object_id, object.epoch);
      const std::int64_t t_ns = rclcpp::Time(tf_msg.header.stamp).nanoseconds();
      auto found = identity_carries_.find(key);
      IdentityCarry carry;
      if (found != identity_carries_.end()) {
        carry = std::move(found->second);
      } else {
        carry = start_carry(object, link_frame, t_ns, grid.resolution);
      }
      add_carry_pose(carry, t_ns, octomap_from_link, grid.resolution);
      if (flush) {
        flush_carry(carry, object, *track, set, base_to_octomap, grid.resolution);
      }
      carries.emplace(key, std::move(carry));
    }
    identity_carries_ = std::move(carries);
    if (grid_cleared > 0) {
      // Declared static geometry is unioned AFTER the clear, so it is never cleared.
      mark_base_points(grid, static_points_);
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "identity clear: %zu grid cell(s) of tracked bindings cleared",
                           grid_cleared);
    }
    if (octree_work) {
      vacate_left_boxes(set, base_to_octomap, live, grid.resolution);
    }
  }

  /// A new binding's carry: its reach about the link, and its poses from `t_attach`
  /// (the attachment's `stamp_ns` on first sight; at most `kIdentityBackfillMaxS`
  /// back) up to `now_ns` from TF history. A pose TF cannot give is skipped.
  IdentityCarry start_carry(const openral_msgs::msg::AttachedCollisionObject& object,
                            const std::string& link_frame, std::int64_t now_ns, double res) {
    IdentityCarry carry;
    std::vector<IdentityPiece> at_link;
    if (place_identity_pieces(object, tf2::Transform::getIdentity(), at_link)) {
      for (const auto& p : at_link) {
        carry.reach_m = std::max(carry.reach_m, p.prim.pose.getOrigin().length() +
                                                    primitive_bounding_radius(p.prim));
      }
    }
    if (object.stamp_ns <= 0 || object.stamp_ns >= now_ns) {
      return carry;
    }
    const auto max_ns = static_cast<std::int64_t>(kIdentityBackfillMaxS * 1e9);
    const auto step_ns = static_cast<std::int64_t>(kIdentityBackfillStepS * 1e9);
    for (std::int64_t t = std::max(object.stamp_ns, now_ns - max_ns); t < now_ns; t += step_ns) {
      try {
        const auto msg = tf_buffer_->lookupTransform(octomap_frame_, link_frame,
                                                     tf2::TimePoint(std::chrono::nanoseconds(t)));
        tf2::Transform pose;
        tf2::fromMsg(msg.transform, pose);
        add_carry_pose(carry, t, pose, res);
      } catch (const tf2::TransformException&) {
        continue;
      }
    }
    return carry;
  }

  /// Queue `pose` (octomap <- link at `t_ns`), interpolated from the last one when the
  /// two are at most `kIdentityMaxInterpGapS` apart.
  static void add_carry_pose(IdentityCarry& carry, std::int64_t t_ns, const tf2::Transform& pose,
                             double res) {
    if (carry.has_last && t_ns == carry.last_ns) {
      return;
    }
    const double gap_s = static_cast<double>(t_ns - carry.last_ns) * 1e-9;
    if (carry.has_last && gap_s > 0.0 && gap_s <= kIdentityMaxInterpGapS) {
      append_swept_poses(carry.pending, carry.last, pose, carry.reach_m, 0.5 * res);
    } else {
      carry.pending.push_back(pose);
    }
    carry.has_last = true;
    carry.last_ns = t_ns;
    carry.last = pose;
  }

  /// Clear from the octree every cell the bound geometry covers at a queued pose.
  void flush_carry(IdentityCarry& carry, const openral_msgs::msg::AttachedCollisionObject& object,
                   const openral_msgs::msg::TrackedObject& track,
                   const openral_msgs::msg::TrackedObjectSet& set,
                   const tf2::Transform& base_to_octomap, double res) {
    std::vector<std::vector<IdentityPiece>> poses;
    SweptBox window;
    for (const auto& pose : carry.pending) {
      std::vector<IdentityPiece> pieces;
      if (!place_identity_pieces(object, pose, pieces)) {
        continue;
      }
      for (const auto& p : pieces) {
        const double r = primitive_bounding_radius(p.prim) + 0.5 * res;
        window.add(p.prim.pose.getOrigin() - tf2::Vector3(r, r, r));
        window.add(p.prim.pose.getOrigin() + tf2::Vector3(r, r, r));
      }
      poses.push_back(std::move(pieces));
    }
    carry.pending.clear();
    IdentityKeepOut keep;
    if (poses.empty() || !identity_keep_out(set, track, base_to_octomap, false, keep)) {
      return;
    }
    send_identity_clears(track, "sweep", window, [&](const tf2::Vector3& c) {
      if (identity_keeps(keep, c, res) || is_static_cell(c)) {
        return false;
      }
      return std::any_of(poses.begin(), poses.end(),
                         [&](const auto& pieces) { return identity_covers(pieces, c, 0.5 * res); });
    });
  }

  /// Vacated cells: a track that has left its last confirmed box — free and confirmed
  /// clear of it, or bound and carried clear of it — gets that box's cells cleared
  /// once, as of now. A re-associated (`identity_inferred`) track vacates nothing.
  void vacate_left_boxes(const openral_msgs::msg::TrackedObjectSet& set,
                         const tf2::Transform& base_to_octomap,
                         const std::map<std::string, std::vector<PayloadPrimitive>>& live,
                         double res) {
    std::map<std::string, IdentityHome> homes;
    for (const auto& track : set.objects) {
      PayloadPrimitive box;
      if (!track_box(track, box)) {
        continue;
      }
      box.pose = base_to_octomap * box.pose;
      const auto home = identity_homes_.find(track.object_id);
      const bool known = home != identity_homes_.end() && home->second.epoch == track.epoch;
      if (track.status == "confirmed") {
        if (known && !track.identity_inferred && clear_of(home->second.box, {box}, res)) {
          vacate(set, track, home->second.box, base_to_octomap, true, res);
        }
        homes[track.object_id] = IdentityHome{track.epoch, box};
        continue;
      }
      if (!known) {
        continue;
      }
      const auto carried = live.find(track.object_id);
      if (track.status == "bound" && carried != live.end() &&
          clear_of(home->second.box, carried->second, res)) {
        vacate(set, track, home->second.box, base_to_octomap, false, res);
        continue;  // spent
      }
      homes.emplace(track.object_id, home->second);
    }
    identity_homes_ = std::move(homes);
  }

  /// Is every primitive more than a cell clear of `home`?
  static bool clear_of(const PayloadPrimitive& home, const std::vector<PayloadPrimitive>& prims,
                       double res) {
    const tf2::Transform local = home.pose.inverse();
    return std::all_of(prims.begin(), prims.end(), [&](const PayloadPrimitive& p) {
      return primitive_surface_distance(home, local * p.pose.getOrigin()) >
             primitive_bounding_radius(p) + res;
    });
  }

  void vacate(const openral_msgs::msg::TrackedObjectSet& set,
              const openral_msgs::msg::TrackedObject& track, const PayloadPrimitive& home,
              const tf2::Transform& base_to_octomap, bool keep_own_box, double res) {
    IdentityKeepOut keep;
    if (!identity_keep_out(set, track, base_to_octomap, keep_own_box, keep)) {
      return;
    }
    SweptBox window;
    add_payload_to_swept_box(window, {home}, tf2::Transform::getIdentity());
    const tf2::Transform local = home.pose.inverse();
    send_identity_clears(track, "vacated", window, [&](const tf2::Vector3& c) {
      return primitive_surface_distance(home, local * c) <= 0.0 && !identity_keeps(keep, c, res) &&
             !is_static_cell(c);
    });
  }

  bool is_static_cell(const tf2::Vector3& c) const {
    return static_keys_.count(octree_->coordToKey(octomap::point3d(
               static_cast<float>(c.x()), static_cast<float>(c.y()), static_cast<float>(c.z())))) !=
           0;
  }

  void send_identity_clears(const openral_msgs::msg::TrackedObject& track, const char* what,
                            const SweptBox& window,
                            const std::function<bool(const tf2::Vector3&)>& clear) {
    bool truncated = false;
    const auto boxes =
        octree_clear_boxes(*octree_, window, clear, kIdentityMaxClearBoxes, truncated);
    if (boxes.empty()) {
      return;
    }
    if (!sweep_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "identity clear: %s not available; %s's cells stay in the octree",
                           sweep_client_->get_service_name(), track.object_id.c_str());
      return;
    }
    sweep_client_->prune_requests_older_than(std::chrono::system_clock::now() -
                                             kSweepRequestTimeout);
    for (const auto& box : boxes) {
      auto req = std::make_shared<octomap_msgs::srv::BoundingBoxQuery::Request>();
      req->min.x = box.lo.x();
      req->min.y = box.lo.y();
      req->min.z = box.lo.z();
      req->max.x = box.hi.x();
      req->max.y = box.hi.y();
      req->max.z = box.hi.z();
      sweep_client_->async_send_request(req);
    }
    RCLCPP_INFO(this->get_logger(),
                "identity clear (%s): %s epoch %u, %zu clear_bbox cell box(es) in %s%s", what,
                track.object_id.c_str(), track.epoch, boxes.size(), octomap_frame_.c_str(),
                truncated ? "; TRUNCATED, the rest stay in the octree" : "");
  }

  std::string base_frame_;
  std::string octomap_frame_;
  double resolution_{0.05};
  double coverage_center_[3]{0.0, 0.0, 0.5};
  double coverage_radius_m_{0.0};
  std::string static_points_file_;
  std::vector<tf2::Vector3> static_points_;
  bool attached_clear_enabled_{true};
  double attached_clear_padding_m_{0.0};
  double attach_sweep_padding_m_{0.05};
  double attached_state_timeout_s_{0.5};
  double max_octree_age_s_{1.0};
  rclcpp::Time octree_received_;
  rclcpp::Time octree_stamp_;
  bool octree_stale_{false};
  AttachSweepLedger attach_sweep_ledger_;
  rclcpp::Client<octomap_msgs::srv::BoundingBoxQuery>::SharedPtr sweep_client_;
  static constexpr std::chrono::seconds kSweepRequestTimeout{5};
  double sweep_period_s_{0.2};
  std::vector<SweptBox> swept_boxes_;
  SweepSupports sweep_supports_;
  std::set<std::string> sweep_support_keys_;
  std::set<std::pair<std::string, std::uint64_t>> swept_seen_;
  rclcpp::Time last_sweep_clear_{0, 0, RCL_ROS_TIME};
  SupportPatchAnchors support_patch_anchors_;
  std::map<std::pair<std::string, std::uint32_t>, IdentityCarry> identity_carries_;
  std::map<std::string, IdentityHome> identity_homes_;
  octomap::KeySet static_keys_;
  rclcpp::Time last_identity_flush_{0, 0, RCL_ROS_TIME};
  AttachLinkTfFrames attach_link_tf_frames_;
  std::unordered_set<std::string> unmapped_links_warned_;  ///< one hint per link, not per tick

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

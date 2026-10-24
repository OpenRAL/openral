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
// It also clears tracked attachments by identity. The bridge removes occupancy
// only for an unambiguous tracked-object binding with a usable tracker record;
// attachment geometry, support witnesses, and link transforms alone do not
// prove that nearby occupied cells belong to the payload.
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
    // there is nothing left to choose. This parameter is retained for launch
    // compatibility; `on_timer` warns if the map disagrees.
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

    // Identity-based attached-object clearing. On by default; geometric attachment
    // primitives without a validated tracked-object binding leave occupancy intact.
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
    attached_state_timeout_s_ = this->declare_parameter<double>("attached_state_timeout_s", 0.5);
    // Optional octree clearing is used only by the identity-validated tracker path below.
    const auto sweep_service =
        this->declare_parameter<std::string>("payload_sweep_clear_service", "");
    sweep_period_s_ = this->declare_parameter<double>("payload_sweep_clear_period_s", 0.2);
    if (!sweep_service.empty()) {
      sweep_client_ = this->create_client<octomap_msgs::srv::BoundingBoxQuery>(sweep_service);
      RCLCPP_WARN(this->get_logger(), "tracked identity sweep clearing ON (every %.2f s) via %s",
                  sweep_period_s_, sweep_service.c_str());
    }
    // Release window (issue #418 F2): how long a just-RELEASED track's box may be cleared.
    // MUST equal the safety kernel's `tracked_release_timeout_s` (K5, default 3.0 s):
    // longer clears the object's cells after the kernel's window closed; shorter re-marks
    // them under the frozen payload while the hand is still exempt against it. Same
    // bound as the kernel's, (0, 10]; anything else refuses to start.
    released_clear_timeout_s_ =
        this->declare_parameter<double>("released_clear_timeout_s", kReleasedClearTimeoutS);
    if (!(released_clear_timeout_s_ > 0.0) ||
        !(released_clear_timeout_s_ <= kReleasedClearTimeoutMaxS)) {
      throw std::invalid_argument("released_clear_timeout_s must be in (0, 10]");
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
      // Not fatal — the grid takes the octree's lattice regardless. The
      // resolution parameter is retained for launch compatibility only.
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "octree resolution %g != `resolution` parameter %g; the grid uses the "
                           "octree's",
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
      // Declared geometry can add occupancy, never authorize removal.
      const std::size_t added = mark_base_points(grid, static_points_);
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "declared static geometry added %zu cells the camera map lacked", added);
    }
    clearing_complete_ = true;
    clear_by_identity(grid, base_to_octomap);
    const rclcpp::Time produced = this->now();
    grid.header.stamp = produced;
    grid.source_stamp = octree_stamp_;
    grid.attachment_revision = certified_attachment_revision();
    voxel_pub_->publish(grid);
    // The one number that says how far behind the world this grid is: capture
    // of the newest cloud in it to now. Throttled; the kernel carries the same
    // age on every check span as `safety.world_voxel_data_age_ms`.
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "world_voxels data age %.0f ms at publish (octree received %.0f ms ago)",
                         (produced - octree_stamp_).seconds() * 1e3, octree_age_s * 1e3);
  }

  /// The attachment revision this grid certifies (`OccupancyVoxels.attachment_revision`):
  /// the revision of the fresh attachment state the identity-clearing pass processed,
  /// provided every bound identity it had to clear was handled (a missing TF or unusable
  /// tracker set withholds it: 0). This is a processing acknowledgement, not proof that every
  /// attached object's cells disappeared or that an asynchronous octree clear completed.
  /// Attachments without an unambiguous tracked-object binding remain obstacles and do not
  /// withhold it.
  std::uint64_t certified_attachment_revision() const {
    const auto state = world_state_;
    if (!attached_clear_enabled_ || state == nullptr || !clearing_complete_) {
      return 0;
    }
    const double age_s = (this->now() - rclcpp::Time(state->header.stamp)).seconds();
    return age_s > attached_state_timeout_s_ ? 0 : state->attachment_revision;
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
      for (const auto& object : state->attached_objects) {
        if (object.evidence_kind == kTrackedObjectEvidence) {
          clearing_complete_ = false;  // a tracker binding this tick could not clear
        }
      }
      identity_carries_.clear();
      identity_homes_.clear();
      identity_was_bound_.clear();
      release_clears_.clear();
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
    // Before this tick's bindings replace the last tick's: a window opens on the tick a
    // track the bridge was clearing turns up RELEASED.
    const auto released =
        advance_release_clears(release_clears_, identity_was_bound_, *state, base_frame_,
                               now.nanoseconds(), released_clear_timeout_s_);
    IdentityKeys was_bound;
    std::map<std::pair<std::string, std::uint32_t>, IdentityCarry> carries;
    std::map<std::string, std::vector<PayloadPrimitive>> live;  // bound, octomap frame
    std::size_t grid_cleared = 0;
    for (const auto& object : state->attached_objects) {
      const char* why = "";
      const auto* track = identity_clearing_track(object, *state, base_frame_, &why);
      if (track == nullptr) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "identity clear: attachment %s evidence '%s' has no unique valid "
                             "tracked-object binding (reason=%s); occupancy stays mapped",
                             object.object_id.c_str(), object.evidence_kind.c_str(), why);
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
        clearing_complete_ = false;
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
        clearing_complete_ = false;
        continue;
      }
      grid_cleared += clear_identity_cells(grid, pieces, keep);
      was_bound.emplace(object.object_id, object.epoch);
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
    identity_was_bound_ = std::move(was_bound);
    for (const auto* track : released) {
      grid_cleared += clear_released_box(grid, *track, set, base_to_octomap, flush);
    }
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

  /// Release window (issue #418 F2): the RELEASED track's box ⊕ half a cell, on the grid
  /// every tick and (on a flush) in the octree, with the identity exclusions. While it is
  /// open the kernel's frozen payload (K5) stands in for the object; without this its
  /// own re-marked cells meet that payload and stop the robot. Returns grid cells cleared.
  std::size_t clear_released_box(openral_msgs::msg::OccupancyVoxels& grid,
                                 const openral_msgs::msg::TrackedObject& track,
                                 const openral_msgs::msg::TrackedObjectSet& set,
                                 const tf2::Transform& base_to_octomap, bool flush) {
    std::vector<IdentityPiece> pieces;
    IdentityKeepOut keep;
    if (!released_clear_pieces(set, track, tf2::Transform::getIdentity(), pieces, keep)) {
      return 0;
    }
    const std::size_t cleared = clear_identity_cells(grid, pieces, keep);
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "identity clear (released): %s epoch %u, %zu grid cell(s) of its "
                         "release box cleared",
                         track.object_id.c_str(), track.epoch, cleared);
    if (!flush || !released_clear_pieces(set, track, base_to_octomap, pieces, keep)) {
      return cleared;
    }
    const double res = grid.resolution;
    const PayloadPrimitive& box = pieces.front().prim;
    const double r = primitive_bounding_radius(box) + 0.5 * res;
    SweptBox window;
    window.add(box.pose.getOrigin() - tf2::Vector3(r, r, r));
    window.add(box.pose.getOrigin() + tf2::Vector3(r, r, r));
    send_identity_clears(track, "released", window, [&](const tf2::Vector3& c) {
      return identity_covers(pieces, c, 0.5 * res) && !identity_keeps(keep, c, res) &&
             !is_static_cell(c);
    });
    return cleared;
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
  double attached_state_timeout_s_{0.5};
  double max_octree_age_s_{1.0};
  rclcpp::Time octree_received_;
  rclcpp::Time octree_stamp_;
  bool octree_stale_{false};
  rclcpp::Client<octomap_msgs::srv::BoundingBoxQuery>::SharedPtr sweep_client_;
  static constexpr std::chrono::seconds kSweepRequestTimeout{5};
  double sweep_period_s_{0.2};
  std::map<std::pair<std::string, std::uint32_t>, IdentityCarry> identity_carries_;
  std::map<std::string, IdentityHome> identity_homes_;
  IdentityKeys identity_was_bound_;  ///< bindings cleared by identity on the last tick
  ReleaseClears release_clears_;
  double released_clear_timeout_s_{kReleasedClearTimeoutS};
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
  /// False once any clearing step of the grid being built withheld what it owed.
  bool clearing_complete_{true};
  rclcpp::Publisher<openral_msgs::msg::OccupancyVoxels>::SharedPtr voxel_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace openral_octomap_bridge

// Copyright 2026 OpenRAL Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Test-only exemption providers (ADR-0121) for test_exemption_provider_kernel. Built only with
// BUILD_TESTING and never installed: the test serves them from a temporary ament prefix, so no
// install tree ever carries a provider that tolerates every pair.

#include "openral_safety_kernel/exemption_provider.hpp"

#include <cstdint>
#include <stdexcept>

#include <pluginlib/class_list_macros.hpp>

#include "recorder_log.hpp"

namespace openral_safety_kernel_test {

namespace osk = openral_safety_kernel;

class Base : public osk::ExemptionProvider {
public:
  bool initialize(const osk::CollisionModel& /*model*/,
                  rclcpp::node_interfaces::NodeParametersInterface::SharedPtr /*params*/,
                  const rclcpp::Logger& /*logger*/) override {
    return true;
  }
};

/// The most permissive provider: tolerates every pair it is asked about.
class TolerateEveryPair : public Base {
public:
  std::uint32_t api_version() const noexcept override { return osk::kExemptionProviderApiVersion; }
  const char* name() const noexcept override { return "tolerate-every-pair/1"; }
  osk::ProviderVerdict voxel_pair(const osk::VoxelPairQuery& /*q*/) noexcept override {
    return osk::ProviderVerdict::kTolerate;
  }
  osk::ProviderVerdict self_pair(const osk::SelfPairQuery& /*q*/) noexcept override {
    return osk::ProviderVerdict::kTolerate;
  }
  osk::ProviderVerdict payload_voxel_pair(const osk::PayloadPairQuery& /*q*/) noexcept override {
    return osk::ProviderVerdict::kTolerate;
  }
  osk::ProviderVerdict payload_self_pair(const osk::PayloadPairQuery& /*q*/) noexcept override {
    return osk::ProviderVerdict::kTolerate;
  }
};

/// Refuses everything and records what the kernel hands it (API v2 view and hooks). Its counters
/// are read by the tests through `osk_test_recorder()`, which they link.
class Recorder : public Base {
public:
  std::uint32_t api_version() const noexcept override { return osk::kExemptionProviderApiVersion; }
  const char* name() const noexcept override { return "recorder/1"; }
  void begin_candidate(const osk::CandidateView& view) noexcept override;
  void end_candidate(const osk::CandidateOutcome& outcome) noexcept override;
  void on_reset(osk::ResetReason why) noexcept override;
  osk::ProviderVerdict voxel_pair(const osk::VoxelPairQuery& q) noexcept override;
};

/// The Recorder, holding every cover the kernel offers (API v3).
class Holder : public Recorder {
public:
  const char* name() const noexcept override { return "holder/1"; }
  osk::ProviderVerdict hold_cover(const osk::CoverHoldQuery& /*q*/) noexcept override {
    ++osk_test_recorder()->hold_asked;
    return osk::ProviderVerdict::kTolerate;
  }
};

/// Releases a bound payload once the tracker has let go of any object (API v3), the in-tree K5's
/// by-tracker rule. The freeze and its window are the kernel's.
class Releaser : public Recorder {
public:
  const char* name() const noexcept override { return "releaser/1"; }
  osk::ProviderVerdict release_payload(const osk::ReleaseQuery& q) noexcept override {
    for (std::size_t k = 0; k < q.n_tracked; ++k) {
      if (q.tracked[k].released && q.tracked[k].epoch == q.epoch) {
        return osk::ProviderVerdict::kTolerate;
      }
    }
    return osk::ProviderVerdict::kRefuse;
  }
};

/// A minimal #349-style rule (API v3 test): a masked hand vs a cell is tolerated while the
/// kernel's measured hand speed and raw lead are within bounds; asks for the kernel's brake on the
/// same bounds (the lag rig's 0.02 m/s and 0.05 m).
class Braker : public Recorder {
public:
  const char* name() const noexcept override { return "braker/1"; }
  osk::BrakeConfig brake_config() const noexcept override {
    osk::BrakeConfig b;
    b.enabled = true;
    b.max_speed_mps = 0.02;
    b.max_lead_m = 0.05;
    return b;
  }
  osk::ProviderVerdict voxel_pair(const osk::VoxelPairQuery& q) noexcept override {
    return q.hand_speed_mps <= 0.02 && q.hand_lead_m <= 0.05 ? osk::ProviderVerdict::kTolerate
                                                             : osk::ProviderVerdict::kRefuse;
  }
};

/// The Recorder, replacing the open ADR-0115 region as the grasp exemption source.
class RecorderReplacing : public Recorder {
public:
  const char* name() const noexcept override { return "recorder-replacing/1"; }
  bool replaces_grasp_region() const noexcept override { return true; }
};

/// Built against another interface version.
class WrongApiVersion : public TolerateEveryPair {
public:
  std::uint32_t api_version() const noexcept override {
    return osk::kExemptionProviderApiVersion + 1;
  }
};

/// Declines at initialize.
class Declines : public TolerateEveryPair {
public:
  bool initialize(const osk::CollisionModel& /*model*/,
                  rclcpp::node_interfaces::NodeParametersInterface::SharedPtr /*params*/,
                  const rclcpp::Logger& /*logger*/) override {
    return false;
  }
};

/// Throws at initialize.
class Throws : public TolerateEveryPair {
public:
  bool initialize(const osk::CollisionModel& /*model*/,
                  rclcpp::node_interfaces::NodeParametersInterface::SharedPtr /*params*/,
                  const rclcpp::Logger& /*logger*/) override {
    throw std::runtime_error("provider initialize failure");
  }
};

}  // namespace openral_safety_kernel_test

extern "C" openral_safety_kernel_test::RecorderLog* osk_test_recorder() {
  static openral_safety_kernel_test::RecorderLog log;
  return &log;
}

namespace openral_safety_kernel_test {

void Recorder::begin_candidate(const osk::CandidateView& view) noexcept {
  RecorderLog& log = *osk_test_recorder();
  ++log.begins;
  log.measured = view.measured != nullptr;
  log.declaration_live = view.declaration.live;
  log.grasp_region = view.declaration.grasp_region != nullptr;
  log.hands = view.declaration.hands != nullptr;
  log.region_max_age_ns = view.region_max_age_ns;
  log.workspace = view.declaration.workspace != nullptr;
  log.grid_seq = view.grid_seq;
  log.workspace_period_s = view.workspace_period_s;
  log.stream_fresh = view.tracked_stream_fresh;
  log.n_tracked = static_cast<int>(view.n_tracked);
  log.cover_ok0 = view.n_tracked > 0 && view.tracked[0].cover_ok;
  log.n_attachments = static_cast<int>(view.n_attachments);
  if (view.n_attachments > 0) {
    log.att0_evidence = static_cast<int>(view.attachments[0].evidence);
    log.att0_track_index = view.attachments[0].track_index;
    log.att0_on_hand = view.attachments[0].on_hand;
    log.att0_bind_low_z = view.attachments[0].bind_low_z;
  }
  log.attached_model_set = view.attached_model != nullptr;
  log.static_points_declared = view.static_points_declared;
}

osk::ProviderVerdict Recorder::voxel_pair(const osk::VoxelPairQuery& q) noexcept {
  RecorderLog& log = *osk_test_recorder();
  ++log.voxel_asked;
  log.hand_speed_mps = q.hand_speed_mps;
  log.hand_lead_m = q.hand_lead_m;
  log.row = q.row;
  return osk::ProviderVerdict::kRefuse;
}

void Recorder::end_candidate(const osk::CandidateOutcome& outcome) noexcept {
  RecorderLog& log = *osk_test_recorder();
  ++log.ends;
  log.horizon = outcome.horizon;
  log.n_dof = outcome.n_dof;
  log.q_last = outcome.q_last_row != nullptr && outcome.n_dof > 0 ? outcome.q_last_row[0] : 0.0;
}

void Recorder::on_reset(osk::ResetReason why) noexcept {
  RecorderLog& log = *osk_test_recorder();
  ++log.resets;
  log.last_reset = static_cast<int>(why);
}

}  // namespace openral_safety_kernel_test

PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Recorder,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Holder, openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Releaser,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Braker, openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::RecorderReplacing,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::TolerateEveryPair,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::WrongApiVersion,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Declines,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Throws, openral_safety_kernel::ExemptionProvider)

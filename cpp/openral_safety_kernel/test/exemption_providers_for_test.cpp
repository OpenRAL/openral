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
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::RecorderReplacing,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::TolerateEveryPair,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::WrongApiVersion,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Declines,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Throws, openral_safety_kernel::ExemptionProvider)

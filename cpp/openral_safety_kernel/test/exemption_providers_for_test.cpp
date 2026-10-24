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

PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::TolerateEveryPair,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::WrongApiVersion,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Declines,
                       openral_safety_kernel::ExemptionProvider)
PLUGINLIB_EXPORT_CLASS(openral_safety_kernel_test::Throws, openral_safety_kernel::ExemptionProvider)

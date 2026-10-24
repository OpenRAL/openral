// Copyright 2026 OpenRAL Contributors
// SPDX-License-Identifier: Apache-2.0
//
// ADR-0121 exemption-provider loader, on the real lifecycle kernel. Every refusal must leave the
// kernel configured and exactly as with no provider (fail closed, never a configure failure), and
// be announced in `exemption_provider_status()` (also on /diagnostics). The success path loads a
// real pluginlib class from the never-installed test library, served from a temporary ament
// prefix built here.

#include "openral_safety_kernel/lifecycle_kernel.hpp"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>

#include "provider_test_prefix.hpp"

namespace osk = openral_safety_kernel;

namespace {

class ExemptionProviderKernelTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { osk_test::serve_test_providers(); }
  void SetUp() override {
    setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "OFF", 1);
    rclcpp::init(0, nullptr);
  }
  void TearDown() override { rclcpp::shutdown(); }

  // A 2-link collision model (link names "link0", "link1"), world voxels on, plus the provider
  // parameters under test.
  static rclcpp::NodeOptions options(const std::string& provider, const std::string& hal_mode,
                                     const std::vector<std::string>& links) {
    rclcpp::NodeOptions opts;
    opts.parameter_overrides({
        {"n_dof", std::int64_t{2}},
        {"joint_position_min", std::vector<double>{-3.14, -3.14}},
        {"joint_position_max", std::vector<double>{3.14, 3.14}},
        {"joint_velocity_max", std::vector<double>{3.15, 3.15}},
        {"joint_torque_max", std::vector<double>{5.0, 5.0}},
        {"self_collision_enabled", true},
        {"self_collision_margin_m", 0.0},
        {"collision_n_links", std::int64_t{2}},
        {"collision_parent", std::vector<std::int64_t>{-1, 0}},
        {"collision_joint_kind", std::vector<std::int64_t>{1, 1}},
        {"collision_dof_index", std::vector<std::int64_t>{0, 1}},
        {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0}},
        {"collision_axis", std::vector<double>{0, 0, 1, 0, 1, 0}},
        {"collision_capsule_link", std::vector<std::int64_t>{0, 1}},
        {"collision_capsule_radius", std::vector<double>{0.05, 0.05}},
        {"collision_capsule_half_length", std::vector<double>{0.15, 0.15}},
        {"collision_capsule_origin_xyzrpy",
         std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"collision_allowed_pairs", std::vector<std::int64_t>{0, 1}},
        {"collision_link_names", std::vector<std::string>{"link0", "link1"}},
        {"hal_mode", hal_mode},
        {"exemption_provider", provider},
        {"exemption_provider_links", links},
    });
    return opts;
  }

  static std::shared_ptr<osk::SafetyKernelLifecycleNode> configured(const std::string& name,
                                                                    const rclcpp::NodeOptions& o) {
    auto node = std::make_shared<osk::SafetyKernelLifecycleNode>(name, o);
    const rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED,
                                         "unconfigured");
    EXPECT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS)
        << "a provider refusal must never fail configure";
    return node;
  }

  static std::string cls(const char* c) { return osk_test::test_provider(c); }
};

bool starts_with(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }

}  // namespace

TEST_F(ExemptionProviderKernelTest, NoProviderByDefault) {
  auto node = configured("osk_prov_none", options("", "sim", {}));
  EXPECT_FALSE(node->exemption_provider_loaded());
  EXPECT_EQ(node->exemption_provider_status(), "none");
}

TEST_F(ExemptionProviderKernelTest, LoadsARealPluginInSimWithAValidMask) {
  auto node = configured("osk_prov_ok", options(cls("TolerateEveryPair"), "sim", {"link1"}));
  EXPECT_TRUE(node->exemption_provider_loaded());
  EXPECT_EQ(node->exemption_provider_status(), "loaded: tolerate-every-pair/1");
  const rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE,
                                         "inactive");
  EXPECT_EQ(node->on_cleanup(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_FALSE(node->exemption_provider_loaded()) << "cleanup unloads the provider";
  EXPECT_EQ(node->exemption_provider_status(), "none");
}

TEST_F(ExemptionProviderKernelTest, OffSimIsRefusedWhateverTheProvider) {
  for (const char* mode : {"real", "", "SIM", "sim "}) {
    auto node = configured(std::string("osk_prov_real_") + std::to_string(std::string(mode).size()),
                           options(cls("TolerateEveryPair"), mode, {"link1"}));
    EXPECT_FALSE(node->exemption_provider_loaded()) << "hal_mode '" << mode << "'";
    EXPECT_TRUE(starts_with(node->exemption_provider_status(), "refused: sim only"))
        << node->exemption_provider_status();
  }
}

TEST_F(ExemptionProviderKernelTest, AnEmptyOrUnknownMaskIsRefused) {
  auto empty = configured("osk_prov_nolinks", options(cls("TolerateEveryPair"), "sim", {}));
  EXPECT_FALSE(empty->exemption_provider_loaded());
  EXPECT_TRUE(starts_with(empty->exemption_provider_status(), "refused: exemption_provider_links"))
      << empty->exemption_provider_status();
  auto unknown =
      configured("osk_prov_badlink", options(cls("TolerateEveryPair"), "sim", {"link1", "arm9"}));
  EXPECT_FALSE(unknown->exemption_provider_loaded());
  EXPECT_NE(unknown->exemption_provider_status().find("unknown link: arm9"), std::string::npos)
      << unknown->exemption_provider_status();
}

TEST_F(ExemptionProviderKernelTest, AMissingPluginIsRefused) {
  auto node = configured("osk_prov_missing", options(cls("NoSuchProvider"), "sim", {"link1"}));
  EXPECT_FALSE(node->exemption_provider_loaded());
  EXPECT_TRUE(starts_with(node->exemption_provider_status(), "refused: load failed"))
      << node->exemption_provider_status();
}

TEST_F(ExemptionProviderKernelTest, AnApiVersionMismatchIsRefused) {
  auto node = configured("osk_prov_api", options(cls("WrongApiVersion"), "sim", {"link1"}));
  EXPECT_FALSE(node->exemption_provider_loaded());
  EXPECT_TRUE(starts_with(node->exemption_provider_status(), "refused: api_version"))
      << node->exemption_provider_status();
}

TEST_F(ExemptionProviderKernelTest, AProviderThatDeclinesOrThrowsIsRefused) {
  auto declines = configured("osk_prov_declines", options(cls("Declines"), "sim", {"link1"}));
  EXPECT_FALSE(declines->exemption_provider_loaded());
  EXPECT_EQ(declines->exemption_provider_status(), "refused: initialize declined");
  auto throws = configured("osk_prov_throws", options(cls("Throws"), "sim", {"link1"}));
  EXPECT_FALSE(throws->exemption_provider_loaded());
  EXPECT_TRUE(starts_with(throws->exemption_provider_status(), "refused: initialize threw"))
      << throws->exemption_provider_status();
}

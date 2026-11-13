// SPDX-License-Identifier: Apache-2.0
// See guarded_inference_controller.hpp for what is guarded and why.

#include "openral_hal_unitree/guarded_inference_controller.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

#include <rclcpp/rclcpp.hpp>

#include "isaac_deploy_core/inference_controller/config_parser.hpp"
#include "isaac_deploy_core/inference_controller/input/input_builder.hpp"
#include "isaac_ros_deploy_converters/converters/message_to_tensor_converter.hpp"
#include "isaac_ros_deploy_ros2_control/converters/state_interface_converter.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace openral_hal_unitree {

namespace {
constexpr char kTimeoutParam[] = "topic_input_timeout_ms";
constexpr char kSourcesParam[] = "topic_input_sources";

using CallbackReturn = controller_interface::CallbackReturn;

/// `name` as a string parameter, declared with `fallback` when absent — the same rule
/// upstream applies to `source_to_topic.<source>` / `source_message_type.<source>`, so
/// both read the same value.
std::string string_param(const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
                         const std::string& name, const std::string& fallback) {
  if (!node->has_parameter(name)) {
    node->declare_parameter<std::string>(name, fallback);
  }
  return node->get_parameter(name).as_string();
}

/// source -> kind of every input the v4.6 base reads from a ROS topic. Mirrors upstream's
/// classification in InferenceController::load_config: not feedback, not a hardware kind.
std::optional<std::map<std::string, std::string>> topic_input_kinds(const std::string& config_path,
                                                                    std::string& error) {
  try {
    const YAML::Node yaml = YAML::LoadFile(config_path);
    const auto graph = isaac_deploy_core::parse_graph_config(yaml);
    if (!graph.has_value()) {
      error = std::string(graph.error().message);
      return std::nullopt;
    }
    const auto sections = isaac_deploy_core::merge_graph_to_model_config(*graph, yaml);
    if (!sections.has_value()) {
      error = std::string(sections.error().message);
      return std::nullopt;
    }
    const auto inputs =
        isaac_deploy_core::InputBuilder::Config::create_from_model_config(*sections);
    if (!inputs.has_value()) {
      error = std::string(inputs.error().message);
      return std::nullopt;
    }
    isaac_ros_deploy_ros2_control::initialize_state_interface_converters();
    const auto& state_registry =
        isaac_ros_deploy_ros2_control::StateInterfaceConverterRegistry::instance();
    std::map<std::string, std::string> kinds;
    for (const auto& term : inputs->terms) {
      if (term.output_key.empty() && !state_registry.contains(term.kind)) {
        kinds.emplace(term.source, term.kind);
      }
    }
    return kinds;
  } catch (const std::exception& e) {
    error = e.what();
    return std::nullopt;
  }
}
}  // namespace

std::optional<std::string> topic_input_timeout_refusal(double timeout_ms) {
  // Bounded both ways so the nanosecond conversion can neither round a set timeout to 0
  // nor overflow (either would disarm the watchdog while the log says it is armed).
  if (timeout_ms == 0.0 ||
      (timeout_ms >= kMinTopicInputTimeoutMs && timeout_ms <= kMaxTopicInputTimeoutMs)) {
    return std::nullopt;
  }
  std::ostringstream reason;
  reason << kTimeoutParam << " is " << timeout_ms << "; it must be 0 (no watchdog) or within ["
         << kMinTopicInputTimeoutMs << ", " << kMaxTopicInputTimeoutMs << "] ms";
  return reason.str();
}

std::optional<std::string> topic_input_sources_refusal(const std::vector<std::string>& listed,
                                                       const std::set<std::string>& topic_sources) {
  for (const auto& source : listed) {
    if (topic_sources.count(source) == 0) {
      return std::string(kSourcesParam) + " lists '" + source +
             "', which this isaac_ros_deploy (v4.6) does not read from a topic: it is not a "
             "policy input, or its kind is read from hardware state interfaces. v5.0 would "
             "subscribe to it; this controller cannot, so it refuses rather than ignore it";
    }
  }
  return std::nullopt;
}

CallbackReturn GuardedInferenceController::on_init() {
  const auto result = InferenceController::on_init();
  if (result != CallbackReturn::SUCCESS) {
    return result;
  }
  try {
    // Kept when the YAML already set them (auto_declare does not overwrite).
    auto_declare<double>(kTimeoutParam, 0.0);
    auto_declare<std::vector<std::string>>(kSourcesParam, std::vector<std::string>{});
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn
GuardedInferenceController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  const auto node = get_node();
  watched_.clear();
  timeout_ns_ = 0;

  // Everything is validated before the base loads (and warms up) the model.
  double timeout_ms = 0.0;
  std::vector<std::string> listed;
  try {
    timeout_ms = node->get_parameter(kTimeoutParam).as_double();
    listed = node->get_parameter(kSourcesParam).as_string_array();
  } catch (const rclcpp::ParameterTypeException& e) {
    RCLCPP_ERROR(node->get_logger(),
                 "Refusing to configure: %s must be a double (ms) and %s a string array: %s",
                 kTimeoutParam, kSourcesParam, e.what());
    return CallbackReturn::ERROR;
  }
  if (const auto refusal = topic_input_timeout_refusal(timeout_ms)) {
    RCLCPP_ERROR(node->get_logger(), "Refusing to configure: %s", refusal->c_str());
    return CallbackReturn::ERROR;
  }

  std::string error;
  const auto kinds = topic_input_kinds(node->get_parameter("config_path").as_string(), error);
  if (!kinds.has_value()) {
    // The base reports the same config error with its own wording; refuse here so the
    // watchdog is never configured against a config it could not read.
    RCLCPP_ERROR(node->get_logger(), "Refusing to configure: cannot read the LEAPP config: %s",
                 error.c_str());
    return CallbackReturn::ERROR;
  }
  std::set<std::string> topic_sources;
  for (const auto& [source, kind] : *kinds) {
    topic_sources.insert(source);
  }
  if (const auto refusal = topic_input_sources_refusal(listed, topic_sources)) {
    RCLCPP_ERROR(node->get_logger(), "Refusing to configure: %s", refusal->c_str());
    return CallbackReturn::ERROR;
  }

  // One subscription per topic, as upstream groups them.
  std::unordered_map<std::string, std::string> type_by_topic;
  if (timeout_ms > 0.0) {
    isaac_ros_deploy_converters::initialize_input_converters();
    auto& registry = isaac_ros_deploy_converters::MessageToTensorConverterRegistry::instance();
    for (const auto& [source, kind] : *kinds) {
      const auto converter = registry.create_for_kind(
          kind, source, string_param(node, "source_message_type." + source, ""));
      if (!converter) {
        RCLCPP_ERROR(node->get_logger(),
                     "Refusing to configure: topic input '%s' (kind '%s') has no converter; "
                     "v4.6 feeds it zeros and the watchdog cannot watch it",
                     source.c_str(), kind.c_str());
        return CallbackReturn::ERROR;
      }
      const auto topic = string_param(node, "source_to_topic." + source, source);
      const auto [it, inserted] = type_by_topic.emplace(topic, converter->get_message_type());
      if (!inserted && it->second != converter->get_message_type()) {
        // Upstream logs this and skips the second source, which then runs on zeros.
        RCLCPP_ERROR(node->get_logger(),
                     "Refusing to configure: topic '%s' carries both '%s' and '%s' (source "
                     "'%s'); v4.6 would feed that source zeros",
                     topic.c_str(), it->second.c_str(), converter->get_message_type().c_str(),
                     source.c_str());
        return CallbackReturn::ERROR;
      }
    }
  }

  const auto result = InferenceController::on_configure(previous_state);
  if (result != CallbackReturn::SUCCESS) {
    return result;
  }

  timeout_ns_ = static_cast<int64_t>(std::llround(timeout_ms * 1e6));
  for (const auto& [topic, type] : type_by_topic) {
    auto watched = std::make_unique<WatchedTopic>();
    watched->topic = topic;
    // Shared, not raw: a callback still running on the executor after a re-configure or an
    // unload freed `watched` keeps its counter alive.
    watched->subscription = node->create_generic_subscription(
        topic, type, rclcpp::QoS(10),
        [received = watched->received](std::shared_ptr<rclcpp::SerializedMessage>) {
          received->fetch_add(1, std::memory_order_relaxed);
        });
    watched_.push_back(std::move(watched));
    RCLCPP_INFO(node->get_logger(), "Watching '%s' [%s]: stale after %.1f ms", topic.c_str(),
                type.c_str(), timeout_ms);
  }
  return result;
}

CallbackReturn
GuardedInferenceController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  // A message from before this activation never counts: every topic must deliver anew.
  for (auto& watched : watched_) {
    watched->seen = watched->received->load(std::memory_order_relaxed);
    watched->fresh_at_ns.reset();
  }
  return InferenceController::on_activate(previous_state);
}

bool GuardedInferenceController::topic_inputs_are_fresh(int64_t now_ns) {
  bool fresh = true;
  for (auto& watched : watched_) {
    const uint64_t received = watched->received->load(std::memory_order_relaxed);
    if (received != watched->seen) {
      watched->seen = received;
      watched->fresh_at_ns = now_ns;
    }
    // A message first seen on THIS tick does not count yet: the base controller's own
    // subscription may not have stored it, and the policy would step on the one before.
    if (!watched->fresh_at_ns.has_value() || *watched->fresh_at_ns >= now_ns ||
        now_ns - *watched->fresh_at_ns > timeout_ns_) {
      fresh = false;
    }
  }
  return fresh;
}

controller_interface::return_type
GuardedInferenceController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  if (timeout_ns_ > 0 && !topic_inputs_are_fresh(time.nanoseconds())) {
    bool all_written = true;
    for (auto& command : command_interfaces_) {
      all_written = command.set_value(std::numeric_limits<double>::quiet_NaN()) && all_written;
    }
    if (!all_written) {
      // Retried next tick; the reference keeps its last value for this one.
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                            "Could not invalidate every policy command interface this tick");
    }
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                         "A topic input is stale (none within %s); invalidating the policy's "
                         "commands so safety_controller holds the pose captured at stale entry",
                         kTimeoutParam);
    return controller_interface::return_type::OK;
  }
  return InferenceController::update(time, period);
}

}  // namespace openral_hal_unitree

PLUGINLIB_EXPORT_CLASS(openral_hal_unitree::GuardedInferenceController,
                       controller_interface::ControllerInterface)

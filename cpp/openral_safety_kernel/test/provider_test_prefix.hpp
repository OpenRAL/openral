// Copyright 2026 OpenRAL Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Serves the never-installed test exemption providers (ADR-0121) to pluginlib from a temporary
// ament prefix: the package marker, the plugin-description index entry for the base class's
// package, the package.xml pluginlib names the description's package from, the description and
// the library. Shared by test_exemption_provider_kernel and test_lifecycle_kernel.

#ifndef OPENRAL_SAFETY_KERNEL_TEST__PROVIDER_TEST_PREFIX_HPP_
#define OPENRAL_SAFETY_KERNEL_TEST__PROVIDER_TEST_PREFIX_HPP_

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

namespace osk_test {

inline constexpr const char* kTestPkg = "openral_safety_kernel_test_providers";

/// `<test package>/<class>`, the pluginlib name of one test provider.
inline std::string test_provider(const char* cls) { return std::string(kTestPkg) + "/" + cls; }

/// Builds the prefix once per process and prepends it to AMENT_PREFIX_PATH. Idempotent.
inline void serve_test_providers() {
  namespace fs = std::filesystem;
  static bool served = false;
  if (served) {
    return;
  }
  const fs::path root =
      fs::temp_directory_path() / ("osk_provider_prefix_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const fs::path index = root / "share" / "ament_index" / "resource_index";
  fs::create_directories(index / "packages");
  fs::create_directories(index / "openral_safety_kernel__pluginlib__plugin");
  fs::create_directories(root / "share" / kTestPkg);
  fs::create_directories(root / "lib");
  std::ofstream(index / "packages" / kTestPkg).close();
  std::ofstream(index / "openral_safety_kernel__pluginlib__plugin" / kTestPkg)
      << "share/" << kTestPkg << "/plugins.xml";
  std::ofstream(root / "share" / kTestPkg / "package.xml")
      << "<?xml version=\"1.0\"?>\n<package format=\"3\"><name>" << kTestPkg
      << "</name><version>0.0.0</version><description>test</description>"
      << "<maintainer email=\"t@t\">t</maintainer><license>Apache-2.0</license></package>\n";
  std::ofstream xml(root / "share" / kTestPkg / "plugins.xml");
  xml << "<library path=\"test_exemption_providers\">\n";
  for (const char* cls : {"TolerateEveryPair", "WrongApiVersion", "Declines", "Throws", "Recorder",
                          "RecorderReplacing"}) {
    xml << "  <class name=\"" << kTestPkg << "/" << cls
        << "\" type=\"openral_safety_kernel_test::" << cls
        << "\" base_class_type=\"openral_safety_kernel::ExemptionProvider\"/>\n";
  }
  xml << "</library>\n";
  xml.close();
  const fs::path lib(TEST_PROVIDERS_LIB);
  fs::create_symlink(lib, root / "lib" / lib.filename());
  const char* old = std::getenv("AMENT_PREFIX_PATH");
  const std::string value =
      root.string() + (old != nullptr && *old != '\0' ? std::string(":") + old : "");
  setenv("AMENT_PREFIX_PATH", value.c_str(), 1);
  served = true;
}

}  // namespace osk_test

#endif  // OPENRAL_SAFETY_KERNEL_TEST__PROVIDER_TEST_PREFIX_HPP_

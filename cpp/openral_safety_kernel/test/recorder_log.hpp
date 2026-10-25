// Copyright 2026 OpenRAL Contributors
// SPDX-License-Identifier: Apache-2.0
//
// What the `Recorder` test exemption provider (ADR-0121) saw, shared between the never-installed
// test provider library and the tests that link it.

#ifndef OPENRAL_SAFETY_KERNEL_TEST__RECORDER_LOG_HPP_
#define OPENRAL_SAFETY_KERNEL_TEST__RECORDER_LOG_HPP_

#include <cstddef>

namespace openral_safety_kernel_test {

/// What the `Recorder` test provider saw (exemption_providers_for_test.cpp).
struct RecorderLog {
  int begins{0};
  int ends{0};
  int resets{0};
  int last_reset{-1};
  bool measured{false};
  bool declaration_live{false};
  bool grasp_region{false};
  bool hands{false};
  long long region_max_age_ns{0};
  int horizon{0};
  std::size_t n_dof{0};
  double q_last{0.0};
  bool workspace{false};
  unsigned long long grid_seq{0};
  double workspace_period_s{0.0};
  bool stream_fresh{false};
  int voxel_asked{0};
  double hand_speed_mps{0.0};
  double hand_lead_m{0.0};
  int row{0};
  bool cover_ok0{false};  ///< the first tracked view's cover_ok at the last begin_candidate
  int n_tracked{0};
  int hold_asked{0};
  int n_attachments{0};
  int att0_evidence{-1};
  int att0_track_index{-2};
  bool att0_on_hand{false};
  bool static_points_declared{false};
  double att0_bind_low_z{0.0};
  bool attached_model_set{false};
};

}  // namespace openral_safety_kernel_test

extern "C" openral_safety_kernel_test::RecorderLog* osk_test_recorder();

#endif  // OPENRAL_SAFETY_KERNEL_TEST__RECORDER_LOG_HPP_

// SPDX-License-Identifier: Apache-2.0
// ADR-0020 / CLAUDE.md §5.2: no exceptions across the kernel boundary.
// Use std::expected-shaped Result type to propagate validator outcomes.

#pragma once

#include <optional>
#include <utility>
#include <variant>

namespace openral_safety_kernel {

/// Compact std::expected-shaped error wrapper for the validator path.
///
/// The validator runs on the hot path (per /openral/candidate_action
/// message) and must not allocate. We return `Result<void, Violation>`
/// by value; `Violation` is a small POD-ish struct holding the enum kind
/// + fixed-size string buffers so no `std::string` allocation happens on
/// the rejection path either (see validator.hpp).
template <typename T, typename E>
class Result {
public:
  /// Construct an ok result holding `T`.
  static Result ok(T value) { return Result(std::move(value)); }
  /// Construct an error result holding `E`.
  static Result err(E error) { return Result(std::move(error), error_tag{}); }

  bool has_value() const noexcept { return payload_.index() == 0; }
  bool has_error() const noexcept { return payload_.index() == 1; }
  explicit operator bool() const noexcept { return has_value(); }

  const T& value() const& noexcept { return std::get<0>(payload_); }
  T& value() & noexcept { return std::get<0>(payload_); }

  const E& error() const& noexcept { return std::get<1>(payload_); }
  E& error() & noexcept { return std::get<1>(payload_); }

private:
  struct error_tag {};
  explicit Result(T value) : payload_(std::in_place_index<0>, std::move(value)) {}
  Result(E error, error_tag) : payload_(std::in_place_index<1>, std::move(error)) {}

  std::variant<T, E> payload_;
};

/// `Result<void, E>` is the common shape for "did the validator pass?".
template <typename E>
class Result<void, E> {
public:
  static Result ok() { return Result(); }
  static Result err(E error) { return Result(std::move(error)); }

  bool has_value() const noexcept { return !error_.has_value(); }
  bool has_error() const noexcept { return error_.has_value(); }
  explicit operator bool() const noexcept { return !error_.has_value(); }

  const E& error() const& noexcept { return *error_; }
  E& error() & noexcept { return *error_; }

private:
  Result() = default;
  explicit Result(E error) : error_(std::in_place, std::move(error)) {}

  // std::optional<E> — small, trivially-destructible when E is a POD.
  std::optional<E> error_;
};

}  // namespace openral_safety_kernel

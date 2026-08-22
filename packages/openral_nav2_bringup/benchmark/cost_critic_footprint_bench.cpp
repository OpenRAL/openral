// Measure what `CostCritic.consider_footprint: true` actually costs on the
// 20 Hz MPPI loop, using the real Jazzy nav2_costmap_2d binaries.
//
// Deliberately OUT of CMakeLists.txt: it is a measurement, not a shipped
// artifact, and it must not put nav2_mppi_controller on the package's build
// path. Build and run it by hand (see this package's README):
//
//   source /opt/ros/jazzy/setup.bash
//   INC=$(for d in /opt/ros/jazzy/include/*/; do echo -n "-I$d "; done)
//   g++ -O2 -std=c++17 benchmark/cost_critic_footprint_bench.cpp \
//       -o /tmp/cost_critic_bench -I/opt/ros/jazzy/include $INC \
//       -L/opt/ros/jazzy/lib -lnav2_costmap_2d_core
//       -Wl,-rpath,/opt/ros/jazzy/lib
//   /tmp/cost_critic_bench
//
// CostCritic::inCollision (jazzy, cost_critic.hpp) does, per sampled point:
//     point cost lookup (always)
//     + collision_checker_.footprintCostAtPose(x, y, theta, footprint)
//       WHEN consider_footprint_ && (cost >= possible_collision_cost_
//                                    || possible_collision_cost_ < 1.0f)
// so the whole delta of flipping the flag is the footprintCostAtPose call.
//
// This times exactly that, on the shipped local costmap geometry, at the two
// real footprint polygons: the manifest chassis, and the chassis grown over
// the payload as measured live (0.860 m forward reach).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_costmap_2d/inflation_layer.hpp"

using nav2_costmap_2d::Costmap2D;
using nav2_costmap_2d::Footprint;

namespace {

// config/nav2_panda_mobile.yaml -> local_costmap
constexpr double kResolution = 0.05;
constexpr unsigned int kCells = 60; // 3 m / 0.05
constexpr double kInflationRadius = 0.40;
constexpr double kFootprintPadding =
    0.01; // Costmap2DROS default, verified live

// config/nav2_panda_mobile.yaml -> FollowPath (MPPI)
constexpr int kBatchSize = 2000;
constexpr int kTimeSteps = 56;
constexpr int kTrajectoryPointStep = 2;
constexpr double kControllerFrequencyHz = 20.0;

Footprint make_footprint(const std::vector<std::pair<double, double>> &xy,
                         double pad) {
  Footprint fp;
  for (const auto &[x, y] : xy) {
    geometry_msgs::msg::Point p;
    // Costmap2DROS::padFootprint pushes each vertex outward by `pad` on each
    // axis.
    p.x = x + std::copysign(pad, x);
    p.y = y + std::copysign(pad, y);
    p.z = 0.0;
    fp.push_back(p);
  }
  return fp;
}

double
time_calls(nav2_costmap_2d::FootprintCollisionChecker<Costmap2D *> &checker,
           const Footprint &fp, int n, std::mt19937 &rng) {
  std::uniform_real_distribution<double> pos(0.8, 2.2);
  std::uniform_real_distribution<double> yaw(-M_PI, M_PI);
  std::vector<std::array<double, 3>> poses(n);
  for (auto &p : poses) {
    p = {pos(rng), pos(rng), yaw(rng)};
  }

  volatile double sink = 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const auto &p : poses) {
    sink += checker.footprintCostAtPose(p[0], p[1], p[2], fp);
  }
  const auto t1 = std::chrono::steady_clock::now();
  (void)sink;
  return std::chrono::duration<double>(t1 - t0).count();
}

double time_point_cost(
    nav2_costmap_2d::FootprintCollisionChecker<Costmap2D *> &checker, int n,
    std::mt19937 &rng) {
  std::uniform_int_distribution<int> cell(2, static_cast<int>(kCells) - 3);
  std::vector<std::array<int, 2>> pts(n);
  for (auto &p : pts) {
    p = {cell(rng), cell(rng)};
  }

  volatile double sink = 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const auto &p : pts) {
    sink += checker.pointCost(p[0], p[1]);
  }
  const auto t1 = std::chrono::steady_clock::now();
  (void)sink;
  return std::chrono::duration<double>(t1 - t0).count();
}

void report(const char *label, const Footprint &fp, double seconds, int n,
            double baseline_ns) {
  const auto [min_d, max_d] = nav2_costmap_2d::calculateMinAndMaxDistances(fp);
  const double per_call_ns = seconds / n * 1e9;
  const long calls =
      static_cast<long>(kBatchSize) * (kTimeSteps / kTrajectoryPointStep);
  const double per_iter_ms = per_call_ns * calls / 1e6;
  const double budget_ms = 1000.0 / kControllerFrequencyHz;
  const double baseline_iter_ms = baseline_ns * calls / 1e6;

  std::printf("\n%s\n", label);
  std::printf("  vertices                    %zu\n", fp.size());
  std::printf("  inscribed radius            %.4f m\n", min_d);
  std::printf("  circumscribed radius        %.4f m\n", max_d);
  std::printf(
      "  inflation_radius %.2f m %s circumscribed -> findCircumscribedCost "
      "returns %s\n",
      kInflationRadius, (kInflationRadius < max_d) ? "<" : ">=",
      (kInflationRadius < max_d)
          ? "0.0  => footprint check on EVERY sampled point"
          : "the inflation cost => footprint check only near obstacles");
  std::printf("  footprintCostAtPose         %.0f ns/call\n", per_call_ns);
  std::printf("  pointCost baseline          %.0f ns/call\n", baseline_ns);
  std::printf(
      "  %ld calls/iteration          %.2f ms  (point-only path: %.2f ms)\n",
      calls, per_iter_ms, baseline_iter_ms);
  std::printf("  delta vs consider_footprint:false   +%.2f ms/iteration\n",
              per_iter_ms - baseline_iter_ms);
  std::printf("  at %.0f Hz (%.0f ms budget): CostCritic alone uses %.0f%% of "
              "the cycle\n",
              kControllerFrequencyHz, budget_ms,
              100.0 * per_iter_ms / budget_ms);
}

} // namespace

int main() {
  Costmap2D costmap(kCells, kCells, kResolution, 0.0, 0.0,
                    nav2_costmap_2d::FREE_SPACE);
  // A realistic local costmap is not empty: fill a band of inflated cost so the
  // lineCost walk hits real values rather than a uniform page.
  std::mt19937 rng(20260822);
  std::uniform_int_distribution<int> cell(0, static_cast<int>(kCells) - 1);
  for (int i = 0; i < 400; ++i) {
    costmap.setCost(cell(rng), cell(rng), nav2_costmap_2d::LETHAL_OBSTACLE);
  }
  for (int i = 0; i < 3000; ++i) {
    costmap.setCost(cell(rng), cell(rng),
                    static_cast<unsigned char>(64 + (i % 128)));
  }

  nav2_costmap_2d::FootprintCollisionChecker<Costmap2D *> checker(&costmap);

  // robots/panda_mobile/robot.yaml -> footprint_polygon
  const Footprint chassis = make_footprint(
      {{0.35, 0.25}, {-0.35, 0.25}, {-0.35, -0.25}, {0.35, -0.25}},
      kFootprintPadding);
  // The live-measured carrying polygon: 0.860 m forward reach
  // (tests/integration/test_nav2_payload_footprint_live.py). Hull of the
  // chassis with a 0.10 m half-extent box at 0.75 m ahead, 0.06 m half-width.
  const Footprint carrying = make_footprint({{0.85, 0.06},
                                             {0.85, -0.06},
                                             {0.35, -0.25},
                                             {-0.35, -0.25},
                                             {-0.35, 0.25},
                                             {0.35, 0.25}},
                                            kFootprintPadding);

  constexpr int kWarmup = 20000;
  constexpr int kSamples = 200000;
  time_calls(checker, chassis, kWarmup, rng);
  time_point_cost(checker, kWarmup, rng);

  const double baseline_ns =
      time_point_cost(checker, kSamples, rng) / kSamples * 1e9;
  const double t_chassis = time_calls(checker, chassis, kSamples, rng);
  const double t_carrying = time_calls(checker, carrying, kSamples, rng);

  std::printf(
      "CostCritic consider_footprint cost, real nav2_costmap_2d (Jazzy)\n"
      "local costmap %ux%u @ %.2f m, MPPI batch %d x steps %d / point_step %d "
      "@ %.0f Hz\n",
      kCells, kCells, kResolution, kBatchSize, kTimeSteps, kTrajectoryPointStep,
      kControllerFrequencyHz);

  report("BASE ONLY (nothing attached)", chassis, t_chassis, kSamples,
         baseline_ns);
  report("CARRYING (payload-grown footprint, 0.860 m reach)", carrying,
         t_carrying, kSamples, baseline_ns);
  return 0;
}

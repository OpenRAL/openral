// SPDX-License-Identifier: Apache-2.0
// Bounded registration for observation-only rigid object models, never a safety
// bound.
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Matrix = std::array<std::array<double, 4>, 4>;

static py::buffer_info matrix_buffer(const py::buffer &input,
                                     py::ssize_t columns) {
  auto p = input.request();
  // Use Python's buffer protocol: Jazzy's pybind11 NumPy adapter predates
  // NumPy 2.
  if (p.ndim != 2 || p.shape[1] != columns || p.itemsize != sizeof(double) ||
      (p.format != py::format_descriptor<double>::format() &&
       p.format != "=d") ||
      p.strides[1] != sizeof(double) ||
      p.strides[0] != columns * sizeof(double)) {
    throw std::invalid_argument(
        "registration needs contiguous float64 matrices");
  }
  return p;
}

static Cloud::Ptr cloud(const py::buffer &points) {
  const auto p = matrix_buffer(points, 3);
  if (p.shape[0] < 4 || p.shape[0] > 8192) {
    throw std::invalid_argument("registration needs 4..8192 XYZ points");
  }
  const auto *values = static_cast<const char *>(p.ptr);
  auto out = pcl::make_shared<Cloud>();
  out->reserve(p.shape[0]);
  for (py::ssize_t i = 0; i < p.shape[0]; ++i) {
    pcl::PointXYZ v;
    for (int j = 0; j < 3; ++j) {
      double value;
      std::memcpy(&value, values + (i * 3 + j) * sizeof(double),
                  sizeof(double));
      if (!std::isfinite(value) || std::abs(value) > 10000.0) {
        throw std::invalid_argument(
            "registration points must be finite metric coordinates");
      }
      v.data[j] = static_cast<float>(value);
    }
    out->push_back(v);
  }
  return out;
}

static std::tuple<Matrix, bool, double, double>
align(const py::buffer &source, const py::buffer &target,
      const py::buffer &initial_buffer, double correspondence_m,
      int iterations) {
  const auto buffer = matrix_buffer(initial_buffer, 4);
  if (buffer.shape[0] != 4) {
    throw std::invalid_argument("initial pose must be 4x4");
  }
  const auto *values = static_cast<const char *>(buffer.ptr);
  Eigen::Matrix4d initial;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double value;
      std::memcpy(&value, values + (i * 4 + j) * sizeof(double),
                  sizeof(double));
      initial(i, j) = value;
    }
  }
  if (!initial.allFinite() ||
      !initial.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-8) ||
      !(initial.block<3, 3>(0, 0).transpose() * initial.block<3, 3>(0, 0))
           .isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
      std::abs(initial.block<3, 3>(0, 0).determinant() - 1.0) > 1e-6 ||
      !std::isfinite(correspondence_m) || correspondence_m <= 0.0 ||
      correspondence_m > 1.0 || iterations < 1 || iterations > 100) {
    throw std::invalid_argument("invalid rigid registration parameters");
  }
  auto src = cloud(source), dst = cloud(target);
  py::gil_scoped_release release;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(src);
  icp.setInputTarget(dst);
  icp.setMaximumIterations(iterations);
  icp.setMaxCorrespondenceDistance(correspondence_m);
  Cloud registered;
  icp.align(registered, initial.cast<float>());
  const Eigen::Matrix4d transform = icp.getFinalTransformation().cast<double>();
  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  tree.setInputCloud(dst);
  std::vector<int> index(1);
  std::vector<float> squared(1);
  double error = 0.0;
  std::size_t count = 0;
  for (const auto &p : registered) {
    if (tree.nearestKSearch(p, 1, index, squared) == 1 &&
        squared[0] <= correspondence_m * correspondence_m) {
      error += squared[0];
      ++count;
    }
  }
  const double rms = count ? std::sqrt(error / count)
                           : std::numeric_limits<double>::infinity();
  Matrix output;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      output[i][j] = transform(i, j);
    }
  }
  return {output, icp.hasConverged(), rms,
          static_cast<double>(count) / src->size()};
}

PYBIND11_MODULE(_openral_pcl_registration, module) {
  module.doc() =
      "Bounded PCL ICP; outputs are observations, not safety certificates";
  module.def("align", &align, py::arg("source"), py::arg("target"),
             py::arg("initial"), py::arg("correspondence_m"),
             py::arg("iterations"));
}

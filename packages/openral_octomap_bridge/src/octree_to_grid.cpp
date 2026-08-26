// SPDX-License-Identifier: Apache-2.0
// OctoMap → dense lattice-aligned occupancy grid (testable core).
//
// The grid's lattice IS the octree's. There are therefore no two lattices to
// reconcile, no relative phase, no relative yaw, and no overlap closure: one
// published cell is one octree cell, located by integer index arithmetic. The
// rotation that used to be dissolved into a per-cell dilation is carried on the
// wire instead, in `OccupancyVoxels.orientation`.
//
// Until 2026-08-25 this rasterized onto a base-frame lattice and marked every
// cell whose cube shared volume with an occupied leaf's. That rule was correct
// and minimal for a base-aligned output — a cell overlapping a leaf might be
// the one holding the surface — but it dilated the obstacle set by 29-35 mm
// median (40 mm worst) on 25 mm cells, which held 48% of the live start-state
// E-stops. See the header and issue #173.

#include "openral_octomap_bridge/octree_to_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <octomap/OcTreeKey.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

namespace openral_octomap_bridge {
namespace {

/// Ceiling on the published grid, well above any kernel's own capacity (the
/// safety kernel's `world_voxel_max_cells` defaults to 589824). This is a
/// crash guard, not a policy: a spec this large is a misconfiguration, and the
/// answer to one is a refusal the node can log, never an allocation attempt.
constexpr std::size_t kMaxCells = 4000000;
constexpr std::uint32_t kMaxCellsPerAxis = 4096;

/// The grid's lattice, in the octree frame.
struct Lattice {
  double resolution{0.0};
  double cell0_min[3]{0.0, 0.0, 0.0};  ///< min corner of cell (0,0,0)
  std::array<std::uint32_t, 3> size{0, 0, 0};
};

/// Build the grid lattice covering `spec`'s ball, snapped onto the octree's own
/// cell boundaries so a leaf lands on whole cells.
///
/// The snap is ARITHMETIC, deliberately, and not `coordToKeyChecked` +
/// `keyToCoord`. Those fail outside the octree's addressable
/// +/-32768*resolution, and a lattice this function declines to place is an
/// empty grid — every obstacle dropped, the one direction this node must never
/// fail in. octomap's own mapping is `keyToCoord(coordToKey(c)) ==
/// (floor(c/res) + 0.5) * res`: the key origin cancels, so the lattice is
/// reproduced exactly here, for every coordinate, without a range to fall out
/// of. (The bbx *query* below still has that range, and still falls back to
/// walking the whole tree when it does.)
bool build_lattice(const octomap::OcTree& tree, const tf2::Transform& base_to_octomap,
                   const GridSpec& spec, Lattice& out) {
  const double res = tree.getResolution();
  if (!(res > 0.0) || !(spec.radius > 0.0) || !std::isfinite(spec.radius)) {
    return false;
  }
  const tf2::Vector3 center_octree =
      base_to_octomap * tf2::Vector3(spec.center[0], spec.center[1], spec.center[2]);
  const double lo[3] = {center_octree.x() - spec.radius, center_octree.y() - spec.radius,
                        center_octree.z() - spec.radius};
  const double hi[3] = {center_octree.x() + spec.radius, center_octree.y() + spec.radius,
                        center_octree.z() + spec.radius};
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(lo[i]) || !std::isfinite(hi[i])) {
      return false;
    }
  }
  // Centre of the octree cell containing the ball's minimum corner.
  const double c0[3] = {(std::floor(lo[0] / res) + 0.5) * res,
                        (std::floor(lo[1] / res) + 0.5) * res,
                        (std::floor(lo[2] / res) + 0.5) * res};

  out.resolution = res;
  for (int i = 0; i < 3; ++i) {
    out.cell0_min[i] = c0[i] - 0.5 * res;
    // Cells needed to reach past `hi`. `+ 1` because index 0 is cell0 itself;
    // the floor keeps a `hi` exactly on a boundary from buying an empty layer.
    const double span = (hi[i] - c0[i]) / res;
    if (!(span < static_cast<double>(kMaxCellsPerAxis))) {
      return false;
    }
    const long n = static_cast<long>(std::floor(span)) + 1;
    if (n <= 0) {
      return false;
    }
    out.size[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(n);
  }
  // The allocation this sizes is the published message. A spec far larger than
  // any kernel would accept must be REFUSED, not attempted: `std::bad_alloc`
  // out of the bridge's timer callback is a crashed perception node, and the
  // caller can only report a grid it did not get.
  const double cells = static_cast<double>(out.size[0]) * static_cast<double>(out.size[1]) *
                       static_cast<double>(out.size[2]);
  return cells <= static_cast<double>(kMaxCells);
}

/// Mark the cells one occupied leaf covers.
///
/// A leaf's edge is the tree resolution or a power-of-two multiple of it, and
/// the lattices are identical, so the leaf covers a whole `k x k x k` block
/// starting at an exact cell boundary. `std::lround` absorbs the float
/// precision of octomap's leaf coordinates (~6e-8 relative), which is orders of
/// magnitude below a cell.
void mark_leaf(const Lattice& lattice, const octomap::point3d& leaf_center, double leaf_size,
               std::vector<std::uint8_t>& occupancy) {
  const double leaf_min[3] = {leaf_center.x() - 0.5 * leaf_size, leaf_center.y() - 0.5 * leaf_size,
                              leaf_center.z() - 0.5 * leaf_size};
  const long k = std::lround(leaf_size / lattice.resolution);
  if (k <= 0) {
    return;
  }
  std::array<long, 3> lo{};
  std::array<long, 3> hi{};
  for (int i = 0; i < 3; ++i) {
    const long first = std::lround((leaf_min[i] - lattice.cell0_min[i]) / lattice.resolution);
    lo[i] = std::max<long>(first, 0);
    hi[i] = std::min<long>(first + k - 1,
                           static_cast<long>(lattice.size[static_cast<std::size_t>(i)]) - 1);
    if (lo[i] > hi[i]) {
      return;  // wholly outside the covered volume
    }
  }
  const std::size_t sx = lattice.size[0];
  const std::size_t sy = lattice.size[1];
  for (long iz = lo[2]; iz <= hi[2]; ++iz) {
    for (long iy = lo[1]; iy <= hi[1]; ++iy) {
      const std::size_t row = sx * (static_cast<std::size_t>(iy) + sy * static_cast<std::size_t>(iz));
      for (long ix = lo[0]; ix <= hi[0]; ++ix) {
        occupancy[row + static_cast<std::size_t>(ix)] = 1;
      }
    }
  }
}

/// The octree key box covering the whole grid, padded by one tree cell so key
/// rounding cannot clip a leaf that grazes it. octomap's bbx iterator descends
/// on the child's full EXTENT overlapping the key box, so a coarse leaf whose
/// centre lies outside the box is still visited.
///
/// False when the box reaches outside the octree's addressable range, where the
/// key conversion fails and a key query would iterate NOTHING at all.
bool grid_box_keys(const octomap::OcTree& tree, const Lattice& lattice, octomap::OcTreeKey& min_key,
                   octomap::OcTreeKey& max_key) {
  const double pad = tree.getResolution();
  const octomap::point3d bbx_min(static_cast<float>(lattice.cell0_min[0] - pad),
                                 static_cast<float>(lattice.cell0_min[1] - pad),
                                 static_cast<float>(lattice.cell0_min[2] - pad));
  const octomap::point3d bbx_max(
      static_cast<float>(lattice.cell0_min[0] + lattice.size[0] * lattice.resolution + pad),
      static_cast<float>(lattice.cell0_min[1] + lattice.size[1] * lattice.resolution + pad),
      static_cast<float>(lattice.cell0_min[2] + lattice.size[2] * lattice.resolution + pad));
  return tree.coordToKeyChecked(bbx_min, min_key) && tree.coordToKeyChecked(bbx_max, max_key);
}

}  // namespace

openral_msgs::msg::OccupancyVoxels rasterize_octree_to_grid(const octomap::OcTree& tree,
                                                            const tf2::Transform& base_to_octomap,
                                                            const GridSpec& spec,
                                                            const std::string& base_frame) {
  openral_msgs::msg::OccupancyVoxels msg;
  msg.header.frame_id = base_frame;
  // An empty grid is the honest answer to "the lattice could not be placed",
  // and the kernel's own size checks reject it rather than reading a stale one.
  msg.orientation.w = 1.0;

  Lattice lattice;
  if (!build_lattice(tree, base_to_octomap, spec, lattice)) {
    return msg;
  }

  const tf2::Transform octomap_to_base = base_to_octomap.inverse();
  const tf2::Vector3 origin_base =
      octomap_to_base *
      tf2::Vector3(lattice.cell0_min[0], lattice.cell0_min[1], lattice.cell0_min[2]);
  const tf2::Quaternion q = octomap_to_base.getRotation().normalized();
  msg.origin.x = origin_base.x();
  msg.origin.y = origin_base.y();
  msg.origin.z = origin_base.z();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
  msg.orientation.w = q.w();
  msg.resolution = lattice.resolution;
  msg.size_x = lattice.size[0];
  msg.size_y = lattice.size[1];
  msg.size_z = lattice.size[2];

  const std::size_t cells = static_cast<std::size_t>(lattice.size[0]) *
                            static_cast<std::size_t>(lattice.size[1]) *
                            static_cast<std::size_t>(lattice.size[2]);
  msg.occupancy.assign(cells, 0);
  if (cells == 0) {
    return msg;
  }

  octomap::OcTreeKey min_key;
  octomap::OcTreeKey max_key;
  if (grid_box_keys(tree, lattice, min_key, max_key)) {
    // Only the leaves that can reach the grid.
    for (auto it = tree.begin_leafs_bbx(min_key, max_key), end = tree.end_leafs_bbx(); it != end;
         ++it) {
      if (tree.isNodeOccupied(*it)) {
        mark_leaf(lattice, it.getCoordinate(), it.getSize(), msg.occupancy);
      }
    }
    return msg;
  }
  // The grid reaches outside the octree's addressable range, where a key-space
  // query would silently iterate NOTHING and drop every obstacle. Walk the
  // whole tree instead: slower, and never wrong in the unsafe direction.
  for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
    if (tree.isNodeOccupied(*it)) {
      mark_leaf(lattice, it.getCoordinate(), it.getSize(), msg.occupancy);
    }
  }
  return msg;
}

}  // namespace openral_octomap_bridge

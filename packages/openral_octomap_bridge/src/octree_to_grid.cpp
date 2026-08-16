// SPDX-License-Identifier: Apache-2.0
// OctoMap → dense base-frame occupancy grid (testable core).
//
// The grid's lattice lives in `base_frame` and the octree's lives in the map /
// odom frame, so the two have an arbitrary relative phase (and, on a mobile
// base, an arbitrary relative yaw). Sampling the octree at each base cell's
// CENTRE — what this function did until 2026-08-16 — therefore quantizes every
// surface to whichever lattice the centre happened to land on, and at a
// half-cell phase that is a whole voxel of displacement toward the robot with
// nothing left where the surface actually is. So the rule is OVERLAP, not
// sampling: a base cell is occupied when its cube shares volume with an
// occupied octree leaf's cube.

#include "openral_octomap_bridge/octree_to_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <octomap/OcTreeKey.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>

namespace openral_octomap_bridge {
namespace {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;

Vec3 to_vec3(const tf2::Vector3& v) { return Vec3{v.x(), v.y(), v.z()}; }

// Overlap is STRICT: two cubes that merely touch (a shared face, edge or
// corner) share no volume and must not mark each other, or a base grid
// phase-aligned with the octree's would mark all 27 cells around every leaf
// instead of the one under it.
//
// "Touching" has to be recognised through octomap's FLOAT leaf coordinates,
// whose ulp is ~6e-8 of the coordinate's own magnitude — three orders of
// magnitude coarser than the double arithmetic around it — so the tolerance is
// relative to how far the leaf sits from the octree origin: 1 µm at the origin,
// 51 µm at 50 m. It stays four orders of magnitude below the half-cell
// penetration that a centre-inside-the-leaf (the pre-2026-08-16 rule) always
// has, which is why no cell the old rule marked can fall out of the new one.
constexpr double kOverlapEps = 1e-6;

// Guards the 9 cross-product axes of the SAT test when the two lattices are
// (near-)parallel and the cross products degenerate to zero vectors. Enlarging
// the projected radii can only make the test report overlap, never separation,
// so it errs toward marking — the conservative direction here.
constexpr double kParallelEps = 1e-9;

/// The two lattices' relative orientation and the grid's cell size, hoisted out
/// of the per-leaf work.
struct GridFrame {
  Mat3 r{};      ///< octree→base rotation
  Mat3 abs_r{};  ///< its entrywise magnitude, padded by `kParallelEps`
  double cell_half{0.0};
};

/// Separating-axis test between the grid's axis-aligned cell cube and an octree
/// leaf cube carried into the base frame by the octree→base rotation.
///
/// `t` is the leaf centre relative to the cell centre in the base frame. Both
/// boxes are cubes, so one half-edge each is enough. `touch_eps` is the
/// strictness tolerance for this leaf.
bool cube_overlaps_cube(const Vec3& t, const GridFrame& frame, double leaf_half, double touch_eps) {
  const Mat3& r = frame.r;
  const Mat3& abs_r = frame.abs_r;
  const double cell_half = frame.cell_half;
  // The grid's own three axes.
  for (int i = 0; i < 3; ++i) {
    const double reach = cell_half + leaf_half * (abs_r[i][0] + abs_r[i][1] + abs_r[i][2]);
    if (std::fabs(t[i]) >= reach - touch_eps) {
      return false;
    }
  }
  // The octree lattice's three axes.
  for (int j = 0; j < 3; ++j) {
    const double reach = leaf_half + cell_half * (abs_r[0][j] + abs_r[1][j] + abs_r[2][j]);
    const double projected = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
    if (std::fabs(projected) >= reach - touch_eps) {
      return false;
    }
  }
  // The 9 edge-cross axes. No strictness tolerance here: an edge-on contact is
  // separated on a cross axis while every face axis still shows real
  // penetration, and subtracting a tolerance from a radius that is legitimately
  // ~0 (parallel lattices) would report separation for boxes that overlap.
  for (int i = 0; i < 3; ++i) {
    const int i1 = (i + 1) % 3;
    const int i2 = (i + 2) % 3;
    for (int j = 0; j < 3; ++j) {
      const int j1 = (j + 1) % 3;
      const int j2 = (j + 2) % 3;
      const double reach =
          cell_half * (abs_r[i1][j] + abs_r[i2][j]) + leaf_half * (abs_r[i][j1] + abs_r[i][j2]);
      const double projected = t[i2] * r[i1][j] - t[i1] * r[i2][j];
      if (std::fabs(projected) > reach) {
        return false;
      }
    }
  }
  return true;
}

/// Mark every cell of `occupancy` whose cube shares volume with one leaf's.
///
/// The leaf can be coarser than the tree resolution — octomap prunes siblings
/// that carry the same value, and a published `/octomap_binary` is pruned — so
/// its own edge length decides its reach, never `tree.getResolution()`.
void mark_cells_overlapping_leaf(const GridSpec& spec, const GridFrame& frame,
                                 const tf2::Transform& octomap_to_base,
                                 const octomap::point3d& leaf_center_octree, double leaf_size,
                                 std::vector<std::uint8_t>& occupancy) {
  const double leaf_half = 0.5 * leaf_size;
  // Scaled to the float precision of this leaf's own coordinate (above).
  const double touch_eps =
      kOverlapEps * (1.0 + static_cast<double>(std::max({std::fabs(leaf_center_octree.x()),
                                                         std::fabs(leaf_center_octree.y()),
                                                         std::fabs(leaf_center_octree.z())})));
  const Vec3 leaf_center =
      to_vec3(octomap_to_base *
              tf2::Vector3(leaf_center_octree.x(), leaf_center_octree.y(), leaf_center_octree.z()));

  // Candidate cells: the base-frame AABB of the (rotated) leaf cube. It
  // over-covers; `cube_overlaps_cube` then decides each candidate exactly.
  const std::array<std::uint32_t, 3> size{spec.sx, spec.sy, spec.sz};
  std::array<long, 3> lo{};
  std::array<long, 3> hi{};
  for (int i = 0; i < 3; ++i) {
    const double reach = leaf_half * (frame.abs_r[i][0] + frame.abs_r[i][1] + frame.abs_r[i][2]);
    lo[i] =
        static_cast<long>(std::floor((leaf_center[i] - reach - spec.box_min[i]) / spec.resolution));
    hi[i] =
        static_cast<long>(std::floor((leaf_center[i] + reach - spec.box_min[i]) / spec.resolution));
    lo[i] = std::max<long>(lo[i], 0);
    hi[i] = std::min<long>(hi[i], static_cast<long>(size[i]) - 1);
    if (lo[i] > hi[i]) {
      return;
    }
  }

  for (long iz = lo[2]; iz <= hi[2]; ++iz) {
    const double cz = spec.box_min[2] + (static_cast<double>(iz) + 0.5) * spec.resolution;
    for (long iy = lo[1]; iy <= hi[1]; ++iy) {
      const double cy = spec.box_min[1] + (static_cast<double>(iy) + 0.5) * spec.resolution;
      for (long ix = lo[0]; ix <= hi[0]; ++ix) {
        const double cx = spec.box_min[0] + (static_cast<double>(ix) + 0.5) * spec.resolution;
        const Vec3 t{leaf_center[0] - cx, leaf_center[1] - cy, leaf_center[2] - cz};
        if (cube_overlaps_cube(t, frame, leaf_half, touch_eps)) {
          const std::size_t idx =
              static_cast<std::size_t>(ix) +
              static_cast<std::size_t>(spec.sx) *
                  (static_cast<std::size_t>(iy) + static_cast<std::size_t>(spec.sy) * iz);
          occupancy[idx] = 1;
        }
      }
    }
  }
}

/// The octree key box covering the whole grid: its 8 corners carried into the
/// octree frame, padded by one tree cell so key rounding cannot clip a leaf
/// that grazes the box. octomap's bbx iterator descends on the child's full
/// EXTENT overlapping the key box, so a coarse leaf whose centre lies outside
/// the box is still visited.
///
/// False when the box reaches outside the octree's addressable
/// ±32768·resolution, where the key conversion fails and the caller must not
/// use a key query (it would iterate nothing at all).
bool grid_box_keys(const octomap::OcTree& tree, const tf2::Transform& base_to_octomap,
                   const GridSpec& spec, octomap::OcTreeKey& min_key, octomap::OcTreeKey& max_key) {
  const Vec3 box_max{spec.box_min[0] + spec.sx * spec.resolution,
                     spec.box_min[1] + spec.sy * spec.resolution,
                     spec.box_min[2] + spec.sz * spec.resolution};
  Vec3 lo{};
  Vec3 hi{};
  for (int corner = 0; corner < 8; ++corner) {
    const Vec3 p =
        to_vec3(base_to_octomap * tf2::Vector3((corner & 1) != 0 ? box_max[0] : spec.box_min[0],
                                               (corner & 2) != 0 ? box_max[1] : spec.box_min[1],
                                               (corner & 4) != 0 ? box_max[2] : spec.box_min[2]));
    for (int i = 0; i < 3; ++i) {
      lo[i] = corner == 0 ? p[i] : std::min(lo[i], p[i]);
      hi[i] = corner == 0 ? p[i] : std::max(hi[i], p[i]);
    }
  }
  const double pad = tree.getResolution();
  const octomap::point3d bbx_min(static_cast<float>(lo[0] - pad), static_cast<float>(lo[1] - pad),
                                 static_cast<float>(lo[2] - pad));
  const octomap::point3d bbx_max(static_cast<float>(hi[0] + pad), static_cast<float>(hi[1] + pad),
                                 static_cast<float>(hi[2] + pad));
  return tree.coordToKeyChecked(bbx_min, min_key) && tree.coordToKeyChecked(bbx_max, max_key);
}

}  // namespace

openral_msgs::msg::OccupancyVoxels rasterize_octree_to_grid(const octomap::OcTree& tree,
                                                            const tf2::Transform& base_to_octomap,
                                                            const GridSpec& spec,
                                                            const std::string& base_frame) {
  openral_msgs::msg::OccupancyVoxels msg;
  msg.header.frame_id = base_frame;
  msg.origin.x = spec.box_min[0];
  msg.origin.y = spec.box_min[1];
  msg.origin.z = spec.box_min[2];
  msg.resolution = spec.resolution;
  msg.size_x = spec.sx;
  msg.size_y = spec.sy;
  msg.size_z = spec.sz;
  const std::size_t cells =
      static_cast<std::size_t>(spec.sx) * static_cast<std::size_t>(spec.sy) * spec.sz;
  msg.occupancy.assign(cells, 0);
  if (cells == 0) {
    return msg;
  }

  const tf2::Transform octomap_to_base = base_to_octomap.inverse();
  GridFrame frame;
  frame.cell_half = 0.5 * spec.resolution;
  for (int i = 0; i < 3; ++i) {
    frame.r[i] = to_vec3(octomap_to_base.getBasis().getRow(i));
    for (int j = 0; j < 3; ++j) {
      frame.abs_r[i][j] = std::fabs(frame.r[i][j]) + kParallelEps;
    }
  }

  octomap::OcTreeKey min_key;
  octomap::OcTreeKey max_key;
  if (grid_box_keys(tree, base_to_octomap, spec, min_key, max_key)) {
    // Only the leaves that can reach the grid box.
    for (auto it = tree.begin_leafs_bbx(min_key, max_key), end = tree.end_leafs_bbx(); it != end;
         ++it) {
      if (tree.isNodeOccupied(*it)) {
        mark_cells_overlapping_leaf(spec, frame, octomap_to_base, it.getCoordinate(), it.getSize(),
                                    msg.occupancy);
      }
    }
    return msg;
  }
  // The box reaches outside the octree's addressable range, where a key-space
  // query would silently iterate NOTHING and drop every obstacle. Walk the
  // whole tree instead: slower, and never wrong in the unsafe direction.
  for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
    if (tree.isNodeOccupied(*it)) {
      mark_cells_overlapping_leaf(spec, frame, octomap_to_base, it.getCoordinate(), it.getSize(),
                                  msg.occupancy);
    }
  }
  return msg;
}

}  // namespace openral_octomap_bridge

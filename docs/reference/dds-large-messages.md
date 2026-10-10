# DDS transport for large messages

The deploy graph moves megabyte samples on best-effort topics: depth clouds
(an Isaac 512×384 depth camera back-projects to ~2.4–2.8 MB of xyz float32), the
robot self-filter's output cloud, depth and colour images. Under ROS 2 Jazzy's
default RMW (Fast DDS 2.14) most of them never arrived. `openral deploy sim` and
`openral deploy run` now export a Fast DDS profile that fixes it; this page is the
evidence and the reasoning.

## Symptom

Isaac deploy sim on Spark (2026-10-04, RMW unset, no profile): the sim sensor
bridge published `/openral/cameras/head_zed/points` at 3.3 Hz (~2.8 MB each), the
robot self-filter emitted `/openral/world_cloud/self_filtered` at 1.9 Hz with gaps
up to 2.0 s, while its own stats counted no drops — the clouds never reached it.
octomap_server consumes the filtered cloud, so the map updated at < 2 Hz, and the
grasp-target leg (which needs a self-filtered cloud within 0.1 s of each depth
frame, `VisionAttachmentBridge.kept_points`) refused most captures as `unfiltered`.

## Root cause

Fast DDS gives every participant (one per ROS process) a **512 KiB shared-memory
segment**, and same-host samples travel through the *publisher's* segment. A sample
larger than that is fragmented into it faster than readers free the fragments; a
best-effort reader gets no repair, so the whole sample is dropped. The cliff sits
exactly at the segment size.

Measured on the reference laptop with `tools/cloud_transport_probe.py` (two
processes, `PointCloud2` xyz float32, 4 Hz, publisher BEST_EFFORT KEEP_LAST 5,
subscriber BEST_EFFORT KEEP_LAST 2 — the sim bridge's and the self-filter's QoS):

| cloud size | delivered |
|---|---|
| 240 KB | 40/40 |
| 480 KB | 40/40 |
| 540 KB | 3/40 |
| 720 KB | 0/40 |
| 1.2 MB | 0/40 |
| 2.8 MB | 0/40 (1/60 on a longer run) |

`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` (what `deploy sim` sets) changes nothing
(0/40 at 2.8 MB). The fraction that survives depends on how fast readers drain the
segment — Spark got roughly half its clouds through, the laptop none — but every
host is on the wrong side of the same cliff. The receive side does not matter: a
publisher with the profile and a subscriber without it delivered 40/40, the
reverse 0/40.

The real ZED on the OpenArm cell publishes a REDUCED cloud (224×128 points ×
16 B ≈ 459 KB), just under the cliff — any larger setting crosses it, and its
HD720 depth and colour frames (2.7–3.7 MB) were above it. `openarm_bench.yaml` has grabbed
VGA since 2026-10-07 (672×376: about 1 MB depth, 1 MB BGRA colour), but those frames still
exceed the cliff, so the options below still apply.

## Options measured (2.8 MB, 4 Hz, same host)

| option | delivered | latency p50 | `publish()` p99 | verdict |
|---|---|---|---|---|
| default Fast DDS, best effort | 0/40 | – | 4 ms | the bug |
| (a) Fast DDS profile, SHM segment 4 / 8 / 16 MiB | 40/40 each | 8 ms | 4 ms | **chosen (16 MiB)** |
| (a) Fast DDS `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA` (± `max_msg_size=4MB`) | 3/40, 2/40 | – | 4 ms | keeps the 512 KiB segment |
| (b) RELIABLE publisher **and** subscriber, default Fast DDS | 40/40 | 19 ms | 4 ms | needs every reader changed |
| (b) RELIABLE publisher, BEST_EFFORT subscriber | 2/40 | – | 4 ms | the reader's QoS decides |
| (d) Cyclone DDS, best effort | 31/40 | 9 ms | 5 ms | lossy too |
| (d) Cyclone DDS, reliable both ends | 40/40 | 9 ms | 5 ms | not the deploy RMW |

Why not the others:

- **(b) RELIABLE** only helps if every *subscriber* asks for it, and a RELIABLE
  reader does not match a BEST_EFFORT writer at all — a camera driver publishing
  best-effort would then deliver nothing. Not generic, and it changes the sensor
  QoS class (CLAUDE.md §2).
- **(c) Downsampling at the source** fixes one topic, not the depth/colour images or
  a vendor driver's cloud, and cannot get under the cliff at the resolution the
  grasp leg needs: a synthetic 512×384 cloud (70° FOV) voxelised at the grasp leg's
  5 mm `mask_without_removed_points` cell is still 0.71 MB (a plane at 1 m) to
  1.06 MB (a 0.5–1.5 m tilted plane); only octomap's 15 mm cell gets it under
  (0.08–0.16 MB), and that would starve the grasp leg's matching.
- **(d) Cyclone** is not the deploy RMW (`_apply_rmw_default` explains why) and was
  lossy best-effort too.

## The fix

`python/cli/src/openral_cli/fastdds_large_data.xml`: the builtin transports
(SHM for same-host peers, UDPv4 for discovery and other hosts) with a 16 MiB
shared-memory segment. `apply_fastdds_large_data_profile` exports it as
`FASTRTPS_DEFAULT_PROFILES_FILE` into the launch environment unless the operator
set a non-empty one, and the `dds_transport_ready:` line names whichever profile the
graph loads (`fastdds_profile=…`, `n/a` on Cyclone/Zenoh). No sudo, no sysctl, no QoS
change: producers stay best-effort and never block.

Through the real `robot_self_filter` (both lossy hops: producer → self-filter →
HAL-style KEEP_LAST 1 reader, 2.8 MB kept whole), on the laptop:

| | 4 Hz | 10 Hz |
|---|---|---|
| default Fast DDS | 0/40 | 0/100 |
| with the profile | 40/40, p50 17 ms | 100/100, p50 14 ms |

`tests/integration/test_large_cloud_transport_live.py` asserts ≥ 95 % and a
non-blocking `publish()` on every live-ROS run.

### Cost and limits

- Fast DDS maps the whole segment into `/dev/shm` for **every** participant,
  resident from creation (measured: an idle subscriber holds 16.8 MB). The graph
  costs ~16 MiB × its process count — about 320 MiB for 20 processes. A container
  with Docker's default 64 MB `/dev/shm` cannot hold more than three; run the deploy
  on the host or with `--ipc=host` / a larger `--shm-size`.
- Only the publisher's segment matters, so the profile must reach the **publishing**
  process. Everything the deploy launch spawns inherits it, scene `drivers:`
  included; a driver started in another shell, or a bare `ros2 launch` of
  `deploy_e2e.launch.py`, needs `FASTRTPS_DEFAULT_PROFILES_FILE` exported by hand.
- Shared memory is same-host only. Across hosts the fragments ride UDP, whose
  receive buffer the profile cannot raise past `net.core.rmem_max` (212992 on Spark)
  without sudo. Not measured here; keep large best-effort producers on the host of
  their consumers.
- Jazzy only: Fast DDS 3 reads `FASTDDS_DEFAULT_PROFILES_FILE`.

## Reproduce

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=88
python tools/cloud_transport_probe.py sub --count 40 --depth 2 &
python tools/cloud_transport_probe.py pub --count 40 --hz 4          # ~0/40
FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/python/cli/src/openral_cli/fastdds_large_data.xml \
  python tools/cloud_transport_probe.py pub --count 40 --hz 4        # 40/40
```

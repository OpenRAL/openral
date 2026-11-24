# DDS transport for large messages

The deploy graph moves megabyte samples on best-effort topics: depth clouds
(an Isaac 512x384 depth camera back-projects to ~2.4-2.8 MB), the robot
self-filter's output cloud, depth and colour images. Under a bare host's
default RMW, Fast DDS 2.14 (ROS 2 Jazzy), most of them never arrived.
`openral deploy sim` and `openral deploy run` therefore export a Fast DDS
profile that fixes it.

**Scope.** The dev and inference images set
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. There the launch skips the profile
(`dds_transport_ready:` prints `fastdds_profile=n/a`) and this page does not
apply; Cyclone best-effort lost 9/40 clouds at 2.8 MB in the measurements below
(row d). The rest of this page covers hosts running Fast DDS.

## Symptom

Without the profile, the sim bridge's depth cloud arrived at ~3 Hz and the
self-filter's output at < 2 Hz with gaps up to 2 s, while its own stats counted
no drops (the clouds never reached it). octomap_server updated at < 2 Hz, and
the grasp-target leg, which needs a self-filtered cloud within 0.1 s of each
depth frame, refused most captures as `unfiltered`.

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
| (d) Cyclone DDS, reliable both ends | 40/40 | 9 ms | 5 ms | needs every reader changed |

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
- **(d) Cyclone** was lossy best-effort too, and is the image default only
  because the images pin it (see Scope).

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

### Service replies

The profile carries `data_writer` profiles for twenty service reply writers:
the `change_state` and `get_state` replies of the ten Nav2 nodes the lifecycle
manager drives, named by reply topic (`rr/<service>Reply`). The RMW waits for
the client's reply reader to be discovered for at most `max_blocking_time`
(default 100 ms) and otherwise drops the reply. On the ~30-process graph
discovery can be slower, and Nav2's lifecycle manager waits on `change_state`
with no timeout, so one dropped reply left Nav2 unconfigured for the run.
These twenty replies may wait 5 s. They are named individually rather than
using the catch-all `service` profile, because the wait runs on the server's
executor thread and would block the safety kernel's single-threaded one.
`tests/unit/test_fastdds_profile_service_scope.py` holds the profile to the
twenty names.

### Localhost discovery reach

With `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` there is no multicast: a
participant announces itself only to the unicast ports of the first N
participant ids on the host, so two participants that both start after the Nth
never find each other. A deploy sim graph is about 50 processes and Nav2 starts
last, so Nav2's nodes did not discover each other. The profile therefore sets
`maxInitialPeersRange` to **120** on the UDP transport (60 one-node processes
all discovered each other).

Not higher, because each domain owns 250 UDP ports and participant id 125 of
domain N is id 0 of domain N+1. A range of 128 announced a graph to the first
participants of the next domain, merging two graphs and making a second
`openral deploy sim` one domain up refuse as an occupied graph; 120 keeps the
last port one below the next domain's first.
`tests/integration/test_adjacent_domain_isolation_live.py` checks this against
the shipped profile. Adjacent domains can still leak under an older OpenRAL or
a custom `FASTRTPS_DEFAULT_PROFILES_FILE` with a range above 120: restart with
the range at 120, or keep the two domains at least two apart. The XML header
comments in `fastdds_large_data.xml` carry the full reasoning.

## Reproduce

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=88
python tools/cloud_transport_probe.py sub --count 40 --depth 2 &
python tools/cloud_transport_probe.py pub --count 40 --hz 4          # ~0/40
FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/python/cli/src/openral_cli/fastdds_large_data.xml \
  python tools/cloud_transport_probe.py pub --count 40 --hz 4        # 40/40
```

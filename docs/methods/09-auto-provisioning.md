# Auto-Provisioning (Detection)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/detect/src/openral_detect/__init__.py`
- `detect_hardware(*, dds_timeout_s=5.0, include=None, exclude=None) -> DetectionReport` — Umbrella probe entry. (in `detect.py:L33`)
- `build_compute_spec(gpu, report) -> ComputeSpec` — Build a `ComputeSpec` from a GPU probe + `DetectionReport`, extracting the same fields `assemble_robot_description` writes into `compute_edge`/`compute_local`, for callers (e.g. `openral doctor`) that only need the compute spec. (in `assemble.py:L83`)
- `assemble_robot_description(detection, *, base_description=None, force_robot_type=None, enrich_cameras=True) -> RobotDescription` — Identify-then-enrich: infers the robot from DDS, then CAN, then USB (strongest evidence first). `force_robot_type` pins the canonical base manifest and raises `ROSConfigError` when it doesn't resolve — e.g. to select SO-100 over the USB-indistinguishable SO-101 default. `enrich_cameras=False` (the interactive wizard's choice) leaves the sensor list for the camera-binding wizard to own; the probe-only `--no-write` path uses `True`. (in `assemble.py:L72`)
- `check_installed_rskills(robot, *, registry_path=None, rskills_dir=None) -> CompatibilityReport` — Walk-all: run `rSkill.check_compatibility` against every installed (and optionally in-tree) skill. (in `compatibility.py:L107`)
- `check_single_rskill(rskill_id, robot) -> CompatibilityReport` — Resolve one id via `load_rskill_manifest` and emit a one-row report with per-section verdicts. (in `compatibility.py:L294`)
- const `PROBE_NAMES: frozenset[str]` — Names accepted by `detect_hardware(include=...)`.

### `python/detect/src/openral_detect/assemble.py`

*Assemble a complete `RobotDescription` from a `DetectionReport`: pick a canonical base manifest (by override, DDS/CAN/USB inference, or a minimal scaffold), then enrich sensors, compute, and ROS 2 metadata. Never raises on a missing catalog entry — falls back to a generic `SensorSpec` and records a warning.*

- `build_compute_spec(gpu, report) -> ComputeSpec` — Build a `ComputeSpec` from a GPU probe + `DetectionReport`, extracting the same fields `assemble_robot_description` writes into `compute_edge`/`compute_local`, for callers (e.g. `openral doctor`) that only need the compute spec. (L59)
- `assemble_robot_description(detection, *, base_description=None, force_robot_type=None, enrich_cameras=True) -> RobotDescription` — Identify-then-enrich; inference order is DDS then CAN then USB, strongest evidence first. `force_robot_type` pins the canonical base manifest and raises `ROSConfigError` when it does not resolve. `enrich_cameras=False` leaves `sensors`/`sensor_bundles` untouched for the camera-binding wizard to own; compute and ROS 2 enrichment always run regardless. (L98)

### `python/detect/src/openral_detect/compatibility.py`
- `class SectionVerdict(BaseModel)` — Per-section verdict for `openral rskill check <rskill_id>`. (L56)
- `class RSkillCompatRow(BaseModel)` — One row in the compatibility report. (L81)
  fields: `repo_id, version, role, manifest_path, embodiment_tags, compatible, reason, failure_kind, sections`
- `class CompatibilityReport(BaseModel)` — `openral rskill check` output. (L97)
  fields: `schema_version, generated_at, robot_name, robot_embodiment_tags, rows`
  - `compatible -> list[RSkillCompatRow]` (property) — Rows that passed every check. (L109)
  - `incompatible -> list[RSkillCompatRow]` (property) — Rows that failed at least one check. (L114)
- `check_installed_rskills(robot, *, registry_path=None, rskills_dir=None) -> CompatibilityReport` — Walk-all: run `rSkill.check_compatibility` against every installed (and optionally in-tree) skill. (L119)
- const `_CLASSIFICATION_KEYWORDS: tuple[tuple[str, FailureKind], ...]` — Keyword → `FailureKind` lookup used by `_classify` to coarsely classify a `ROSCapabilityMismatch` message; most-specific match wins, falls back to `"capability_flag"`. (L227)
- `_evaluate_sections(manifest, robot) -> list[SectionVerdict]` — Run each per-section production check and collect the six verdicts. (L283)
- `check_single_rskill(rskill_id, robot) -> CompatibilityReport` — Resolve `rskill_id` via `load_rskill_manifest` and report its compatibility with `robot` as a one-row report with per-section verdicts; aggregate `compatible` is true when every non-informational section passes. (L365)

### `python/detect/src/openral_detect/detect.py`

*Umbrella entry point `detect_hardware()`: runs every probe sequentially (each wrapped in a `safety_span` for OTel) and aggregates into a single `DetectionReport`. Synchronous and never raises — probe failures append typed warnings to `DetectionReport.warnings` instead.*

- const `PROBE_NAMES: frozenset[str]` — Recognized probe names for `include`/`exclude`: `usb`, `can`, `dds`, `gpu`, `cameras_v4l2`, `cameras_realsense`, `network`. (L34)
- `detect_hardware(*, dds_timeout_s=5.0, include=None, exclude=None) -> DetectionReport` — Probe every supported domain and return a typed report. (L39)

### `python/detect/src/openral_detect/probes/`
- `probe_usb(*, warnings=None) -> UsbProbeResult` — Wraps `openral_cli.autodetect.enumerate_usb_devices` + `match_known_devices`. (usb.py L21)
- `_enrich_can_buses(description, detection) -> RobotDescription` — Replaces the CAN interface names in `hal.parameters.defaults` with the ones actually found on this host, for any robot declaring `hal.parameters.can_bus_bindings` — without it a config could name interfaces that merely look host-derived. A role matching no interface, or more than one, appends a warning and leaves the manifest value alone rather than guessing. Defined in `assemble.py`, not this package.
- `probe_can(*, warnings=None) -> CanProbeResult` — Wraps `openral_cli.autodetect.enumerate_can_interfaces` + `match_can_interfaces`. Finds what USB-serial enumeration structurally can't: a CAN-attached arm (OpenArm) registers a network device, not a `/dev/tty*` node. Read-only — never opens a socket, so probing a live robot can't perturb it. (can.py L85)
- `diagnose_can_matches(matches) -> list[str]` — Turn controller states on matched buses into operator-facing lines: `ERROR-PASSIVE` on an up, correctly-configured bus means frames are leaving the adapter unacknowledged, i.e. the motors are unpowered; `BUS-OFF` points at termination / bitrate. (can.py L47)
- `probe_dds(*, timeout_s=5.0, warnings=None) -> Ros2TopologyResult` — Wraps `scan_dds_topics` + `infer_robot_from_topics` and captures RMW / domain id. (dds.py L19)
- const `NVIDIA_TOPS_BY_NAME_KEYWORD: tuple[tuple[str, float], ...]` — Coarse TOPS estimate keyed by a substring that must appear in the GPU `name` field; first-match-wins, most-specific keyword first. (gpu.py L50)
- const `JETSON_BOARD_TOPS: dict[str, float]` — TOPS per Jetson board (peak INT8, NVIDIA product briefs). AGX Thor is deliberately absent — NVIDIA publishes its headline figure in sparse FP4 TFLOPS with no documented INT8 conversion. (gpu.py L77)
- const `DTYPES_BY_COMPUTE_CAPABILITY: tuple[tuple[tuple[int, int], tuple[QuantizationDtype, ...]], ...]` — Quantization dtypes a CUDA accelerator can reasonably execute, keyed by `(major, minor)` compute capability; conservative — only adds a dtype with hardware backing. (gpu.py L89)
- `probe_gpus(*, warnings=None) -> GpuProbeResult` — NVIDIA via pynvml with an nvidia-smi fallback, Jetson via jtop/proc, Apple Silicon via system_profiler. AGX Thor's compute capability is confirmed (11.0) but it's deliberately excluded from `JETSON_BOARD_TOPS`. On a unified-memory SoC (GB10/DGX Spark, Thor) VRAM falls back to `_unified_memory_vram_fallback` with a shared-with-the-OS warning rather than dropping the GPU. (gpu.py L649)
- `_unified_memory_vram_fallback(warnings, missing_warning, found_warning) -> tuple[int, int] | None` — Shared unified-memory-SoC VRAM fallback for both NVIDIA probes: no discrete VRAM pool → falls back to `_system_memory_mib()` (total RAM + `/proc/meminfo` MemAvailable), appending a warning either way. (gpu.py L301)
- `_probe_cuda_toolkit_version() -> str | None` — prefers `/usr/local/cuda/bin/nvcc` (`_CUDA_HOME_NVCC`) over `$PATH`, so a distro nvcc at `/usr/bin` cannot shadow a newer `/usr/local/cuda-N` install and wrongly close the cuMotion CUDA≥13 gate.
- const `_CUDA_HOME_NVCC: Path` — Canonical "currently selected CUDA toolkit" symlink (`/usr/local/cuda/bin/nvcc`) preferred by `_probe_cuda_toolkit_version` over `$PATH`. (gpu.py L616)
- `_cc_for_jetson_board(board: str) -> tuple[int, int] | None` — Map device-tree board string to CUDA compute capability via `_JETSON_CC_BY_BOARD_KEYWORD` (Thor→11.0, Orin→8.7, Xavier→7.2, Nano→5.3). (gpu.py L213)
- const `_JETSON_CC_BY_BOARD_KEYWORD: tuple[tuple[str, tuple[int, int]], ...]` — Device-tree model keyword → CUDA compute capability table read by `_cc_for_jetson_board`; more-specific keys first. (gpu.py L200)
- `_probe_jetson(warnings, *, model_path=None, release_path=None) -> JetsonInfo | None` — Probe a Tegra host. `model_path` / `release_path` accept fixtures for unit tests; production reads `/proc/device-tree/model` + `/etc/nv_tegra_release`. Returns `None` + warning when the board is unknown. (gpu.py L475)
- const `_DEFAULT_MODEL_PATH: Path` — Default `/proc/device-tree/model` path probed by `_probe_jetson`; overridable for tests. (gpu.py L471)
- const `_DEFAULT_RELEASE_PATH: Path` — `openral_core.TEGRA_RELEASE_PATH`, the default release file probed by `_probe_jetson`; overridable for tests. (gpu.py L472)
- `_probe_nvmm_available(*, search_paths=None) -> bool` — True when `libnvbufsurface.so` is installed (L4T multimedia stack). Populates `RobotCapabilities.nvmm_available`. `search_paths` overrides the canonical roots (`_NVBUFSURFACE_SEARCH_PATHS`) for tests. (gpu.py L238)
- const `_NVBUFSURFACE_SEARCH_PATHS: tuple[Path, ...]` — Canonical L4T install paths for `libnvbufsurface.so` searched by `_probe_nvmm_available`. (gpu.py L232)
- const `_SYSFS_V4L2: str` — `/sys/class/video4linux` — the always-present backend root tried before falling back to `v4l2-ctl --list-devices`. (cameras.py L27)
- `probe_v4l2_cameras(*, warnings=None) -> list[V4l2CameraInfo]` — Linux V4L2 enumeration, one row per physical camera. Prefers the `/sys/class/video4linux` backend (always present, no root) and falls back to `v4l2-ctl --list-devices`; only sysfs exposes the USB descriptor that resolves a `usb_uvc` catalog signature. (cameras.py L30)
- `probe_realsense_devices(*, warnings=None) -> list[RealsenseDeviceInfo]` — `pyrealsense2.context()` wrapper; produces canonical `model_id` ready for catalog reverse-lookup. (realsense.py L15)
- `probe_network(*, warnings=None) -> NetworkProbeResult` — Hostname / per-interface MAC / IPv4 / MTU / link-speed / default route via psutil. (network.py L17)

### `python/detect/src/openral_detect/registry.py`
- const `_WORKSPACE_MARKERS: tuple[str, ...]` — Both `"robots"` and `"python"` must be sibling dirs of a candidate root; a bare `robots/` alone is a plausible name for unrelated content. (L44)
- const `_MAX_ROOT_SEARCH_DEPTH: int` — How many ancestor directories `_discover_workspace_root` walks before giving up. (L45)
- const `_PACKAGE_WORKSPACE_ROOT: Path | None` — Workspace root discovered once at import time by walking up from this module; `None` when no ancestor has both `robots/` and `python/`. (L57)
- const `_OPENRAL_ROBOT_TYPE_TO_DIR: dict[str, str]` — Alias table from a `bh_robot_type` slug to its canonical `robots/<name>/` directory (e.g. `"so100"` → `"so100_follower"`); a slug absent from this table is tried verbatim as a directory name. (L72)
- `canonical_robot_path(bh_robot_type) -> Path | None` — Resolve a robot-type slug (`"so101"`, `"aloha"`, …) to `robots/<name>/robot.yaml`: alias lookup in `_OPENRAL_ROBOT_TYPE_TO_DIR` first, then the slug tried verbatim as a `robots/<slug>/` dir so an operator override can name any committed robot directly. The `robots/` tree is located by walking up to the nearest ancestor holding both `robots/` and `python/`, so resolution doesn't depend on cwd. (L94)
- `signature_for_realsense(model_id) -> SensorSignature` (L138)
- `signature_for_v4l2(name) -> SensorSignature` (L143)
- `signature_for_usb_uvc(vid, pid) -> SensorSignature` (L148)

### `python/detect/src/openral_detect/report.py`

- `class UsbDeviceRecord(BaseModel)` — One USB serial device captured for the report. (L53)
  fields: `port, vid, pid, description`
- `class UsbMatchRecord(BaseModel)` — Detected USB device matched against the VID/PID table. (L66)
  fields: `device, chip, driver_hint, embodiment_tag, bh_robot_type`
- `class UsbProbeResult(BaseModel)` — USB enumeration output. (L76)
  fields: `devices, matches`
- `class CanInterfaceInfo(BaseModel)` — One SocketCAN interface captured for the report. `state` is the field to read when a detected arm will not move — an up, correctly-configured link in `ERROR-PASSIVE` is transmitting into a bus whose motors are unpowered. (L86)
  fields: `name, is_up, fd_enabled, bitrate, data_bitrate, state, driver, mtu, vid, pid, adapter`
- `class CanMatchRecord(BaseModel)` — A group of CAN interfaces whose names identify one known robot (a bimanual arm contributes one per side). (L111)
  fields: `interfaces, chip, driver_hint, embodiment_tag, bh_robot_type`
- `class CanProbeResult(BaseModel)` — SocketCAN enumeration output. (L121)
  fields: `interfaces, matches`
- `class NvidiaGpuInfo(BaseModel)` — One discrete NVIDIA GPU with full attribute set. (L131)
  fields: `index, name, vram_total_mib, vram_free_mib, pci_bus_id, driver_version, cuda_compute_capability, cuda_toolkit_version, tensorrt_version, supported_dtypes, tops_estimate`
- `class JetsonInfo(BaseModel)` — An NVIDIA Jetson SoC. (L152)
  fields: `board, soc, jetpack_version, tops, ram_gb, cuda_compute_capability, cuda_toolkit_version, tensorrt_version, supported_dtypes, power_mode`
- `class AppleSiliconInfo(BaseModel)` — An Apple Silicon SoC. (L167)
  fields: `chip, gpu_cores, unified_mem_gb, supported_dtypes`
- `class GpuProbeResult(BaseModel)` — GPU / SoC discovery output. (L187)
  fields: `nvidia, jetson, apple_silicon, backend`
- `class V4l2CameraInfo(BaseModel)` — One V4L2 camera *device* (not node — a UVC camera registers a capture and a metadata node; `device_path` names the capture one). `vid` / `pid` / `serial` come from the camera's own USB descriptor and are what resolve a `usb_uvc` catalog signature; `0` / `""` for MIPI-CSI cameras and for hosts probed through `v4l2-ctl`. (L199)
  fields: `device_path, name, bus_info, formats, max_resolution, vid, pid, serial`
- `class RealsenseDeviceInfo(BaseModel)` — Intel RealSense device discovered via pyrealsense2. (L222)
  fields: `serial, name, model_id, firmware_version, usb_type`
- `class OrbbecDeviceInfo(BaseModel)` — Orbbec depth camera. (L232)
  fields: `serial, name, model_id, firmware_version`
- `class CameraProbeResult(BaseModel)` — Per-host camera discovery output. (L241)
  fields: `v4l2, realsense, orbbec`
- `class DdsTopicRecord(BaseModel)` — One ROS 2 topic discovered during DDS scan. (L252)
  fields: `name, type_name`
- `class Ros2TopologyResult(BaseModel)` — ROS 2 topology snapshot. (L259)
  fields: `topics, inferred_robot_type, has_robot_description, has_tf, nodes, rmw_implementation, domain_id`
- `class NetworkInterfaceInfo(BaseModel)` — One network interface. (L274)
  fields: `name, mac, ipv4, mtu, link_speed_mbps, is_up`
- `class NetworkProbeResult(BaseModel)` — Per-host network discovery output. (L285)
  fields: `hostname, interfaces, default_route`
- `class DetectionReport(BaseModel)` — Typed result of a single `detect_hardware()` invocation. (L296)
  fields: `schema_version, detected_at, host_os, python_version, usb, can, gpu, cameras, ros2, network, warnings`
  - `derived_runtimes() -> list[RSkillRuntime]` — Translate detected accelerators into a host-supported runtime list. (L331)
  - `derived_dtypes() -> list[QuantizationDtype]` — Union of supported quantization dtypes across detected accelerators. (L364)

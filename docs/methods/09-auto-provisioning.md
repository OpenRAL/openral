# Auto-Provisioning (Detection)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/detect/src/openral_detect/__init__.py`
- `detect_hardware(*, dds_timeout_s=5.0, include=None, exclude=None) -> DetectionReport` — Umbrella probe entry. (in `detect.py:L33`)
- `assemble_robot_description(detection, *, base_description=None, force_robot_type=None, enrich_cameras=True) -> RobotDescription` — Identify-then-enrich. Inference reads DDS, then CAN, then USB — strongest evidence first (a robot naming itself over DDS beats a udev-pinned interface name, which beats a controller chip several arms share). `force_robot_type` (slug or `robots/<name>` dir) pins the canonical base manifest over that inference and raises `ROSConfigError` when it does not resolve — e.g. `--robot so100` to select the older arm, since a bare Feetech plug-in defaults to the SO-101 (the two are USB-indistinguishable). `enrich_cameras` (default `True`) reverse-looks-up each detected camera in the sensor catalog to append it with real intrinsics/FOV/encoding/rate; the CLI's interactive `openral detect` flow passes `enrich_cameras=False` so the camera-binding wizard alone owns the sensor list (the resolved manifest is a template, not the output), while `--no-write`'s probe-only path passes `True` for a non-interactive, self-contained enrichment. `--report <path>` is orthogonal — it only dumps the raw `DetectionReport` JSON and, used without `--no-write`, still runs the full interactive builder with `enrich_cameras=False`. (in `assemble.py:L72`)
- `check_installed_rskills(robot, *, registry_path=None, rskills_dir=None) -> CompatibilityReport` — Walk-all: run `rSkill.check_compatibility` against every installed (and optionally in-tree) skill. (in `compatibility.py:L107`)
- `check_single_rskill(rskill_id, robot) -> CompatibilityReport` — Resolve one id via `load_rskill_manifest` and emit a one-row report with per-section verdicts. (in `compatibility.py:L294`)
- const `PROBE_NAMES: frozenset[str]` — Names accepted by `detect_hardware(include=...)`.

### `python/detect/src/openral_detect/compatibility.py`
- `class SectionVerdict(BaseModel)` — Per-section verdict for `openral rskill check <rskill_id>` (label, compatible, reason, failure_kind, informational). (L56)
- `class RSkillCompatRow(BaseModel)` — One row in the compatibility report. (L81)
  fields: `repo_id, version, role, manifest_path, embodiment_tags, compatible, reason, failure_kind, sections`
- `class CompatibilityReport(BaseModel)` — `openral rskill check` output. (L97)
  fields: `schema_version, generated_at, robot_name, robot_embodiment_tags, rows`
  - `compatible -> list[RSkillCompatRow]` (property)
  - `incompatible -> list[RSkillCompatRow]` (property)
- `_evaluate_sections(manifest, robot) -> list[SectionVerdict]` — Run each per-section production check and collect the six verdicts. (L283)

### `python/detect/src/openral_detect/probes/`
- `probe_usb(*, warnings=None) -> UsbProbeResult` — Wraps `openral_cli.autodetect.enumerate_usb_devices` + `match_known_devices`.
- `_enrich_can_buses(description, detection) -> RobotDescription` — Replaces the CAN interface names in `hal.parameters.defaults` with the ones actually found on this host, for any robot whose manifest declares `hal.parameters.can_bus_bindings`. Without it, `detect` infers the robot *from* the CAN bus and then emits a config naming whichever interfaces the manifest author happened to have — the more dangerous failure, because it looks host-derived and is not. Robot-specific knowledge lives entirely in the manifest (parameter name → role token), so a new CAN robot is a manifest entry rather than a code change; bus count is whatever the robot declares. A role that matches no interface, or more than one, appends to `DetectionReport.warnings` and leaves the manifest value alone rather than guessing which bus is which arm (§1.4).
- `probe_can(*, warnings=None) -> CanProbeResult` — Wraps `openral_cli.autodetect.enumerate_can_interfaces` + `match_can_interfaces`. Finds the transport USB-serial enumeration structurally cannot: a CAN adapter registers a *network* device, so a CAN-attached arm (OpenArm) has no `/dev/tty*` node. Read-only — never opens a socket, so probing a live robot cannot perturb it. Appends the `diagnose_can_matches` lines to `warnings`.
- `diagnose_can_matches(matches) -> list[str]` — Turn controller states on matched buses into operator-facing lines: `ERROR-PASSIVE` on an up, correctly-configured bus means frames are leaving the adapter unacknowledged, i.e. the motors are unpowered; `BUS-OFF` points at termination / bitrate. (can.py)
- `probe_dds(*, timeout_s=5.0, warnings=None) -> Ros2TopologyResult` — Wraps `scan_dds_topics` + `infer_robot_from_topics` and captures RMW / domain id.
- `probe_gpus(*, warnings=None) -> GpuProbeResult` — NVIDIA pynvml → nvidia-smi fallback, Jetson via jtop / proc, Apple Silicon via system_profiler. Includes static `NVIDIA_TOPS_BY_NAME_KEYWORD`, `JETSON_BOARD_TOPS`, `DTYPES_BY_COMPUTE_CAPABILITY`, `_JETSON_CC_BY_BOARD_KEYWORD` tables. AGX Thor is in `_JETSON_CC_BY_BOARD_KEYWORD` (11.0, confirmed against NVML on a real board) but deliberately **not** in `JETSON_BOARD_TOPS` — NVIDIA publishes its headline figure in sparse FP4 TFLOPS with no documented conversion to the peak dense INT8 TOPS the other rows use, so it reports `tops == 0.0` rather than a derived number. Per-device fields degrade independently: on a unified-memory SoC (GB10 / DGX Spark, Thor) NVML answers NOT_SUPPORTED for memory and nvidia-smi prints `[N/A]`, so VRAM falls back to `_system_memory_mib()` (total RAM + `/proc/meminfo` MemAvailable) with a "shared with the OS" warning rather than dropping the GPU.
- `_probe_cuda_toolkit_version() -> str | None` — prefers `/usr/local/cuda/bin/nvcc` (`_CUDA_HOME_NVCC`) over `$PATH`, so a distro nvcc at `/usr/bin` cannot shadow a newer `/usr/local/cuda-N` install and wrongly close the cuMotion CUDA≥13 gate.
- `_cc_for_jetson_board(board: str) -> tuple[int, int] | None` — Map device-tree board string to CUDA compute capability via `_JETSON_CC_BY_BOARD_KEYWORD` (Thor → 11.0, Orin → 8.7, Xavier → 7.2, Maxwell Nano → 5.3); replaces the legacy `"Orin" in board` heuristic. (gpu.py L193)
- `_probe_jetson(warnings, *, model_path=None, release_path=None) -> JetsonInfo | None` — Probe a Tegra host. `model_path` / `release_path` accept fixtures for unit tests; production reads `/proc/device-tree/model` + `/etc/nv_tegra_release`. Returns `None` + warning when the board is unknown. (gpu.py L210)
- `_probe_nvmm_available(*, search_paths=None) -> bool` — True when `libnvbufsurface.so` is installed (L4T multimedia stack). Populates `RobotCapabilities.nvmm_available`. `search_paths` overrides the canonical roots (`_NVBUFSURFACE_SEARCH_PATHS`) for tests. (gpu.py L219)
- `probe_v4l2_cameras(*, warnings=None) -> list[V4l2CameraInfo]` — Linux V4L2 enumeration, one row per *physical* camera. Prefers the `/sys/class/video4linux` backend (always present, no package, no root) and falls back to `v4l2-ctl --list-devices`. Only sysfs exposes the camera's own USB descriptor, which is what resolves a `usb_uvc` catalog signature — a host without `v4l-utils` previously reported zero cameras while several were plugged in.
- `probe_realsense_devices(*, warnings=None) -> list[RealsenseDeviceInfo]` — `pyrealsense2.context()` wrapper; produces canonical `model_id` ready for catalog reverse-lookup.
- `probe_network(*, warnings=None) -> NetworkProbeResult` — Hostname / per-interface MAC / IPv4 / MTU / link-speed / default route via psutil.

### `python/detect/src/openral_detect/registry.py`
- `canonical_robot_path(bh_robot_type) -> Path | None` — Resolve `"so101"` / `"so100"` / `"aloha"` / `"openarm_v2"` / … to `robots/<name>/robot.yaml`. Two-step: alias lookup in `_OPENRAL_ROBOT_TYPE_TO_DIR` (a bare Feetech plug-in resolves to `so101`), then the slug tried verbatim as a `robots/<slug>/` dir — so an operator override can name any committed robot directly (`"so100_follower"`). (L68)
- `signature_for_realsense(model_id) -> SensorSignature` (L112)
- `signature_for_v4l2(name) -> SensorSignature` (L117)
- `signature_for_usb_uvc(vid, pid) -> SensorSignature` (L122)

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

# Auto-Provisioning (Detection)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/detect/src/openral_detect/__init__.py`
- `detect_hardware(*, dds_timeout_s=5.0, include=None, exclude=None) -> DetectionReport` — Umbrella probe entry. (in `detect.py:L33`)
- `assemble_robot_description(detection, *, base_description=None, force_robot_type=None) -> RobotDescription` — Identify-then-enrich. `force_robot_type` (slug or `robots/<name>` dir) pins the canonical base manifest over USB/DDS inference and raises `ROSConfigError` when it does not resolve — e.g. `--robot so100` to select the older arm, since a bare Feetech plug-in defaults to the SO-101 (the two are USB-indistinguishable). (in `assemble.py:L72`)
- `scaffold_robot_environment(description, detection=None) -> RobotEnvironment` — Build a deploy `RobotEnvironment` from a detected robot: pre-fills `robot_id`, HAL serial `port` (detected USB match → manifest default → `/dev/ttyUSB0`), and one `SensorReaderConfig` per camera (with the probed `/dev/video*` when known); leaves `task` / `vla` as `TODO` placeholders and `safety=None` so the robot's own envelope applies. (in `scaffold.py:L51`)
- `check_installed_rskills(robot, *, registry_path=None, rskills_dir=None) -> CompatibilityReport` — Walk-all: run `rSkill.check_compatibility` against every installed (and optionally in-tree) skill. (in `compatibility.py:L107`)
- `check_single_rskill(rskill_id, robot) -> CompatibilityReport` — Resolve one id via `load_rskill_manifest` and emit a one-row report with per-section verdicts. (in `compatibility.py:L294`)
- const `PROBE_NAMES: frozenset[str]` — Names accepted by `detect_hardware(include=...)`.
- const `TODO_TASK_ID` / `TODO_VLA_WEIGHTS_URI: str` — Self-documenting, schema-valid placeholder sentinels written into a scaffolded `RobotEnvironment`; the CLI banner greps for them to warn before deploy. (`scaffold.py`)

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
- `probe_dds(*, timeout_s=5.0, warnings=None) -> Ros2TopologyResult` — Wraps `scan_dds_topics` + `infer_robot_from_topics` and captures RMW / domain id.
- `probe_gpus(*, warnings=None) -> GpuProbeResult` — NVIDIA pynvml → nvidia-smi fallback, Jetson via jtop / proc, Apple Silicon via system_profiler. Includes static `NVIDIA_TOPS_BY_NAME_KEYWORD`, `JETSON_BOARD_TOPS`, `DTYPES_BY_COMPUTE_CAPABILITY`, `_JETSON_CC_BY_BOARD_KEYWORD` tables (ADR-0016).
- `_cc_for_jetson_board(board: str) -> tuple[int, int] | None` — Map device-tree board string to CUDA compute capability via `_JETSON_CC_BY_BOARD_KEYWORD`; replaces the legacy `"Orin" in board` heuristic. ADR-0016 PR 2/3. (gpu.py L193)
- `_probe_jetson(warnings, *, model_path=None, release_path=None) -> JetsonInfo | None` — Probe a Tegra host. `model_path` / `release_path` accept fixtures for unit tests; production reads `/proc/device-tree/model` + `/etc/nv_tegra_release`. Returns `None` + warning when the board is unknown. (gpu.py L210)
- `_probe_nvmm_available(*, search_paths=None) -> bool` — True when `libnvbufsurface.so` is installed (L4T multimedia stack). Populates `RobotCapabilities.nvmm_available`. `search_paths` overrides the canonical roots (`_NVBUFSURFACE_SEARCH_PATHS`) for tests. ADR-0016 PR 2/3. (gpu.py L219)
- `probe_v4l2_cameras(*, warnings=None) -> list[V4l2CameraInfo]` — Linux V4L2 enumeration via `v4l2-ctl --list-devices`.
- `probe_realsense_devices(*, warnings=None) -> list[RealsenseDeviceInfo]` — `pyrealsense2.context()` wrapper; produces canonical `model_id` ready for catalog reverse-lookup.
- `probe_network(*, warnings=None) -> NetworkProbeResult` — Hostname / per-interface MAC / IPv4 / MTU / link-speed / default route via psutil.

### `python/detect/src/openral_detect/registry.py`
- `canonical_robot_path(bh_robot_type) -> Path | None` — Resolve `"so101"` / `"so100"` / `"aloha"` / … to `robots/<name>/robot.yaml`. Two-step: alias lookup in `_OPENRAL_ROBOT_TYPE_TO_DIR` (a bare Feetech plug-in resolves to `so101`), then the slug tried verbatim as a `robots/<slug>/` dir — so an operator override can name any committed robot directly (`"so100_follower"`). (L62)
- `signature_for_realsense(model_id) -> SensorSignature` (L105)
- `signature_for_v4l2(name) -> SensorSignature` (L110)
- `signature_for_usb_uvc(vid, pid) -> SensorSignature` (L115)

### `python/detect/src/openral_detect/report.py`

- `class UsbDeviceRecord(BaseModel)` — One USB serial device captured for the report. (L50)
  fields: `port, vid, pid, description`
- `class UsbMatchRecord(BaseModel)` — Detected USB device matched against the VID/PID table. (L63)
  fields: `device, chip, driver_hint, embodiment_tag, bh_robot_type`
- `class UsbProbeResult(BaseModel)` — USB enumeration output. (L73)
  fields: `devices, matches`
- `class NvidiaGpuInfo(BaseModel)` — One discrete NVIDIA GPU with full attribute set. (L83)
  fields: `index, name, vram_total_mib, vram_free_mib, pci_bus_id, driver_version, cuda_compute_capability, cuda_toolkit_version, tensorrt_version, supported_dtypes, tops_estimate`
- `class JetsonInfo(BaseModel)` — An NVIDIA Jetson SoC. (L104)
  fields: `board, soc, jetpack_version, tops, ram_gb, cuda_compute_capability, cuda_toolkit_version, tensorrt_version, supported_dtypes, power_mode`
- `class AppleSiliconInfo(BaseModel)` — An Apple Silicon SoC. (L119)
  fields: `chip, gpu_cores, unified_mem_gb, supported_dtypes`
- `class GpuProbeResult(BaseModel)` — GPU / SoC discovery output. (L139)
  fields: `nvidia, jetson, apple_silicon, backend`
- `class V4l2CameraInfo(BaseModel)` — One V4L2 camera node. (L151)
  fields: `device_path, name, bus_info, formats, max_resolution`
- `class RealsenseDeviceInfo(BaseModel)` — Intel RealSense device discovered via pyrealsense2. (L161)
  fields: `serial, name, model_id, firmware_version, usb_type`
- `class OrbbecDeviceInfo(BaseModel)` — Orbbec depth camera. (L171)
  fields: `serial, name, model_id, firmware_version`
- `class CameraProbeResult(BaseModel)` — Per-host camera discovery output. (L180)
  fields: `v4l2, realsense, orbbec`
- `class DdsTopicRecord(BaseModel)` — One ROS 2 topic discovered during DDS scan. (L191)
  fields: `name, type_name`
- `class Ros2TopologyResult(BaseModel)` — ROS 2 topology snapshot. (L198)
  fields: `topics, inferred_robot_type, has_robot_description, has_tf, nodes, rmw_implementation, domain_id`
- `class NetworkInterfaceInfo(BaseModel)` — One network interface. (L213)
  fields: `name, mac, ipv4, mtu, link_speed_mbps, is_up`
- `class NetworkProbeResult(BaseModel)` — Per-host network discovery output. (L224)
  fields: `hostname, interfaces, default_route`
- `class DetectionReport(BaseModel)` — Typed result of a single `detect_hardware()` invocation. (L235)
  fields: `schema_version, detected_at, host_os, python_version, usb, gpu, cameras, ros2, network, warnings`
  - `derived_runtimes() -> list[RSkillRuntime]` — Translate detected accelerators into a host-supported runtime list. (L268)
  - `derived_dtypes() -> list[QuantizationDtype]` — Union of supported quantization dtypes across detected accelerators. (L301)


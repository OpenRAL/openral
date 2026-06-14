"""LeRobotDatasetSink — writes LeRobot v3.0 datasets from RolloutRecorder fan-out.

Wraps :class:`lerobot.datasets.LeRobotDataset` (codebase_version ``"3.0"``,
shipped in ``lerobot>=0.5.1``). The sink:

* Lazy-imports lerobot at construction so the package stays importable on
  hosts without lerobot. A typed :class:`ROSConfigError` with the install
  hint is raised the moment ``LeRobotDatasetSink`` is instantiated without
  lerobot in the environment.
* Defers ``LeRobotDataset.create()`` until the first :meth:`write_frame`
  call so the per-camera video shapes can be taken from the actual frame
  (sim and hardware often render at runtime-determined resolutions; the
  ``SensorSpec.intrinsics`` field is optional in the RobotDescription
  schema and not always populated).
* Carries the per-dataset license + repo_id into
  ``meta/info.json["metadata"]`` so downstream consumers can read it
  without consulting an out-of-band metadata table.
* Aggregates a per-dataset ``dataset_success_rate`` across all episodes
  so consumers can filter to successful rollouts without a full pass
  over the rows.

Per CLAUDE.md §1.11 (no mocks) — tests instantiate this sink and exercise
a real :class:`lerobot.datasets.LeRobotDataset` round-trip on hosts where
lerobot is installed; on hosts without lerobot the tests
``pytest.skip`` with the install hint.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog
from openral_core.exceptions import ROSConfigError

from openral_dataset.recorder import DatasetFrame, DatasetSink, EpisodeHeader, EpisodeSummary
from openral_dataset.schema_map import FeatureSpec, features_from_robot

if TYPE_CHECKING:
    from openral_core import RobotDescription

__all__ = ["LeRobotDatasetSink"]

_log = structlog.get_logger(__name__)

# Default license string for produced datasets. The LeRobot convention is
# CC-BY-4.0 (matches the official lerobot/aloha and lerobot/pusht datasets);
# the consent prompt at `openral dataset push` (PR5) enforces an upgrade to a
# more restrictive license when the dataset contains PII.
DEFAULT_LICENSE: Final[str] = "CC-BY-4.0"

# Default video codec for produced MP4s. libsvtav1 is lerobot's v3 default
# (best compression / decode-CPU trade-off); supported by every recent
# ffmpeg build.
DEFAULT_VCODEC: Final[str] = "libsvtav1"

# Tolerance for the round-trip "is fps integer?" check.
_FPS_INTEGER_TOLERANCE: Final[float] = 1e-6

# OTel cross-process correlation columns (ISSUE-109). lerobot v3 maps a
# ``{"dtype": "string", "shape": (1,)}`` feature to a plain
# ``datasets.Value("string")`` parquet column — readable without decoding
# the episode videos. Populated per-frame from the producing ``rskill.tick``
# span so a written row can pivot back into its OTel trace; default ``""``
# when no valid span was in scope.
_TRACE_FEATURES: Final[dict[str, dict[str, Any]]] = {
    "trace_id": {"dtype": "string", "shape": (1,), "names": None},
    "span_id": {"dtype": "string", "shape": (1,), "names": None},
}

# Install hint shown when lerobot is not importable. Includes the workspace
# command (`just sync --all-packages --group sim`) AND the bare-install
# fallback (`uv pip install`) so the error is actionable in both contexts.
_LEROBOT_INSTALL_HINT: Final[str] = (
    "lerobot>=0.5.1 is required for LeRobotDatasetSink. Install it via "
    "the workspace `just sync --all-packages --group metaworld` (or "
    "`--group libero`) or directly via `uv pip install 'lerobot>=0.5.1'`."
)


def _featurespec_to_lerobot_dict(spec: FeatureSpec) -> dict[str, Any]:
    """Project a FeatureSpec onto the v3 features-dict format.

    lerobot v3 expects ``{'dtype': str, 'shape': tuple, 'names': list | None}``
    per feature. The ``names`` field is reserved for per-dimension labels
    (e.g. joint names) which we do not currently surface — sinks may
    upgrade this later by reading ``RobotDescription.joints``.
    """
    return {"dtype": spec.dtype, "shape": tuple(spec.shape), "names": None}


class LeRobotDatasetSink(DatasetSink):
    """LeRobot v3.0 dataset writer fed by :class:`RolloutRecorder`.

    The sink writes one Parquet row per frame and one MP4 chunk per
    episode-camera, exactly the v3 on-disk format. Construction is cheap;
    the underlying :class:`lerobot.datasets.LeRobotDataset` is created
    lazily on the first frame so per-camera image shapes can be taken
    from the actual frame data.

    Args:
        root: Output root directory. Must not already contain a v3
            dataset (lerobot raises a clean error if it does).
        robot: Normative robot description; drives feature shapes via
            :func:`openral_dataset.features_from_robot`.
        fps: Recording cadence in Hz. Locked once the dataset is
            created — every episode must use the same fps.
        repo_id: HF Hub repo id (e.g. ``openral/dataset-pick-cube``).
            Defaults to ``"openral/dataset-<robot_name>"``. Stored on
            disk in ``meta/info.json``; not pushed by this sink (PR5
            owns the push path).
        license: SPDX license string for the produced dataset.
        vcodec: ffmpeg codec for the video streams (default
            ``"libsvtav1"`` — lerobot's v3 default).

    Raises:
        ROSConfigError: If ``lerobot>=0.5.1`` is not importable.
        ROSConfigError: If ``robot.observation_spec`` / ``action_spec``
            are not configured for dataset binding (delegates to
            :func:`features_from_robot`).

    Example:
        >>> from openral_core import RobotDescription
        >>> robot = RobotDescription.from_yaml("robots/so100_follower/robot.yaml")  # doctest: +SKIP
        >>> sink = LeRobotDatasetSink(root="/tmp/ds", robot=robot, fps=30.0)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        root: Path | str,
        robot: RobotDescription,
        fps: float,
        repo_id: str | None = None,
        license: str = DEFAULT_LICENSE,
        vcodec: str = DEFAULT_VCODEC,
        state_shape: tuple[int, ...] | None = None,
        action_dim: int | None = None,
        camera_shape: tuple[int, int] | None = None,
    ) -> None:
        """Initialise the sink and resolve every feature shape up-front.

        Args:
            root: Destination dataset root (must not pre-exist).
            robot: Normative :class:`RobotDescription`; supplies camera
                shapes via ``sensors[*].intrinsics`` and the fallback
                state/action contract from ``observation_spec`` /
                ``action_spec`` when present.
            fps: Recording cadence in Hz (stored as int in
                ``meta/info.json``).
            repo_id: Optional HF Hub repo id (defaults to
                ``openral/dataset-<robot.name>``).
            license: SPDX license string for the produced dataset.
            vcodec: ffmpeg video codec (default ``"libsvtav1"``).
            state_shape: Override for ``observation.state`` shape. Used
                when the sim-specific contract (per ADR-0007) lives on
                the rSkill manifest rather than on
                ``RobotDescription.observation_spec``. The CLI's
                ``_maybe_build_recorder`` passes ``state_contract.dim``
                from the loaded manifest here when present.
            action_dim: Same idea for action dim — overrides
                ``RobotDescription.action_spec.dim`` with the
                manifest's ``action_contract.dim``.
            camera_shape: Override for every camera feature's
                ``(height, width)``. Used by the sim CLI to pass
                ``SceneSpec.observation_height/width`` (sim renders all
                cameras at one scene-level resolution that often
                differs from the physical sensor's intrinsics — pi05's
                RoboCasa eval renders at 128x128 even though
                panda_mobile's USB cameras would natively produce
                256x256). When unset, each camera's shape comes from
                its ``SensorSpec.intrinsics``.

        Per ADR-0019 PR-revert, this sink requires EVERY shape (state,
        action, per-camera HWC) to be declared up-front — there is no
        first-frame fallback. Missing shapes raise
        :class:`ROSConfigError` at construction so a wiring bug
        surfaces before the rollout starts.
        """
        # Lazy-validate lerobot is importable. We don't actually import
        # the symbol yet — that happens at create-time — but failing
        # fast at construction means the wiring code sees the error
        # before a rollout starts.
        try:
            import lerobot  # noqa: F401  # reason: presence probe
        except ImportError as exc:  # reason: keep the chain
            raise ROSConfigError(_LEROBOT_INSTALL_HINT) from exc

        # Build the full features dict NOW from declarations. All shapes
        # must be resolvable from explicit specs (RobotDescription or
        # caller-provided overrides) — no first-frame derivation.
        try:
            feature_specs = features_from_robot(
                robot,
                fps=fps,
                state_shape_override=state_shape,
                action_dim_override=action_dim,
                camera_shape_override=camera_shape,
            )
        except ValueError as exc:
            raise ROSConfigError(
                f"LeRobotDatasetSink: cannot build feature schema for "
                f"robot {robot.name!r}: {exc}. "
                "ADR-0019: state_shape and action_dim must come from either "
                "RobotDescription.{observation_spec,action_spec} (hardware) "
                "or the rSkill manifest's {state_contract,action_contract} "
                "(sim). Camera shapes must come from "
                "SensorSpec.intrinsics.{width,height}."
            ) from exc

        # Validate every declared camera has an explicit shape — no
        # placeholder (0,0,3). The sink writes those shapes verbatim
        # into LeRobotDataset.create()'s features dict, and v3 validates
        # every frame against them.
        for key, spec in feature_specs.items():
            if spec.dtype != "video":
                continue
            if not spec.shape or 0 in spec.shape:
                raise ROSConfigError(
                    f"LeRobotDatasetSink: camera feature {key!r} has shape "
                    f"{spec.shape!r}. ADR-0019 requires every camera's "
                    "intrinsics.{width,height} to be set on the "
                    "RobotDescription.sensors[*] entry that declares the "
                    "matching vla_feature_key."
                )

        self._root = Path(root)
        self._robot = robot
        self._fps = float(fps)
        self._repo_id = repo_id if repo_id is not None else f"openral/dataset-{robot.name}"
        self._license = license
        self._vcodec = vcodec
        self._feature_specs: dict[str, FeatureSpec] = feature_specs

        # Image feature keys are derived once at construction so the
        # write_frame path doesn't recompute them per-frame.
        self._image_keys: tuple[str, ...] = tuple(
            key for key, spec in feature_specs.items() if spec.dtype == "video"
        )

        # Deferred state — filled at the first open_episode.
        self._dataset: Any | None = None  # lerobot.datasets.LeRobotDataset
        self._finalized: bool = False
        self._current_task: str = ""
        self._n_success: int = 0
        self._n_episodes: int = 0
        # ISSUE-109 trace pointers — trace_id of the current episode (first
        # non-empty frame id seen) plus the per-episode (idx → trace_id)
        # accumulator written to meta/ at finalize.
        self._current_episode_idx: int = -1
        self._current_episode_trace: str = ""
        self._episode_traces: list[dict[str, Any]] = []

    # ── DatasetSink protocol ────────────────────────────────────────────────

    def open_episode(self, header: EpisodeHeader) -> None:
        """Begin a new episode. Creates the on-disk dataset on first call.

        ADR-0019: the underlying ``LeRobotDataset`` is created here on
        the FIRST episode (not inside ``write_frame``), so any
        ``LeRobotDataset.create`` error surfaces before any tick has
        run. The features dict is already fully resolved by
        :meth:`__init__` — there is no first-frame fallback.
        """
        if self._finalized:
            raise RuntimeError("LeRobotDatasetSink is finalized; cannot open new episodes")
        if self._dataset is None:
            self._dataset = self._create_dataset()
        self._current_task = header.task_string
        # Reset the per-episode trace capture (filled by the first
        # non-empty frame id; persisted at close_episode).
        self._current_episode_idx = header.episode_idx
        self._current_episode_trace = ""

    def write_frame(self, frame: DatasetFrame) -> None:
        """Append one frame.

        Per-frame state / action / image shapes are validated against
        the declared features dict; mismatched shapes raise
        ``ValueError`` immediately rather than producing a malformed
        dataset.
        """
        if self._dataset is None:
            raise RuntimeError("LeRobotDatasetSink.write_frame called before open_episode")

        # Per-frame shape validation — the declared feature_specs are
        # authoritative; any mismatch is a wiring bug worth surfacing
        # loudly rather than silently corrupting the dataset.
        state_arr = np.asarray(frame.observation_state, dtype=np.float32)
        expected_state_shape = self._feature_specs["observation.state"].shape
        if state_arr.shape != expected_state_shape:
            raise ValueError(
                f"LeRobotDatasetSink: frame state shape {state_arr.shape!r} "
                f"does not match declared feature shape {expected_state_shape!r}"
            )
        action_arr = np.asarray(frame.action, dtype=np.float32)
        expected_action_shape = self._feature_specs["action"].shape
        if action_arr.shape != expected_action_shape:
            raise ValueError(
                f"LeRobotDatasetSink: frame action shape {action_arr.shape!r} "
                f"does not match declared feature shape {expected_action_shape!r}"
            )

        row: dict[str, Any] = {
            "observation.state": state_arr,
            "action": action_arr,
            "task": self._current_task,
            "next.reward": np.asarray([frame.reward], dtype=np.float32),
            "next.done": np.asarray([frame.terminated or frame.truncated], dtype=bool),
            "next.terminated": np.asarray([frame.terminated], dtype=bool),
            "next.truncated": np.asarray([frame.truncated], dtype=bool),
            # next.success is per-frame but encodes episode-level success
            # — filled in at close_episode by way of save_episode's
            # episode_data dict. For per-frame consistency v3 wants a
            # value at every add_frame, so we default to False and the
            # close_episode call publishes the resolved episode success.
            "next.success": np.asarray([False], dtype=bool),
            # ISSUE-109 forward link — plain str (lerobot validates string
            # features with isinstance(value, str)); empty when the
            # producing tick had no valid OTel span.
            "trace_id": frame.trace_id,
            "span_id": frame.span_id,
        }
        for cam_key in self._image_keys:
            stripped = cam_key.removeprefix("observation.images.")
            img = frame.images.get(stripped)
            if img is None:
                raise ValueError(
                    f"LeRobotDatasetSink: frame is missing image '{stripped}' "
                    f"(declared feature {cam_key!r}); available images: "
                    f"{sorted(frame.images.keys())!r}"
                )
            img_arr = np.asarray(img, dtype=np.uint8)
            expected_cam_shape = self._feature_specs[cam_key].shape
            if img_arr.shape != expected_cam_shape:
                raise ValueError(
                    f"LeRobotDatasetSink: camera {cam_key!r} frame shape "
                    f"{img_arr.shape!r} does not match declared feature shape "
                    f"{expected_cam_shape!r}. Update SensorSpec.intrinsics to "
                    "match the actual sensor resolution."
                )
            row[cam_key] = img_arr

        self._dataset.add_frame(row)

        # Capture the episode's trace from the first frame that carries
        # one. trace_id is run-constant (all ticks share the cli.command
        # trace), so the first non-empty id is the episode's trace.
        if not self._current_episode_trace and frame.trace_id:
            self._current_episode_trace = frame.trace_id

    def close_episode(self, summary: EpisodeSummary) -> None:
        """Finalise the episode, encoding video and writing parquet.

        The per-frame ``next.success`` flag is filled at write_frame time
        with ``False``; the *resolved* episode success comes through here
        via :meth:`lerobot.datasets.LeRobotDataset.save_episode`'s
        ``episode_data`` dict, which v3 uses to backfill scalar fields
        post-hoc. The dataset-level ``dataset_success_rate`` is
        recomputed at :meth:`finalize` time.
        """
        # Record the episode → trace_id pointer (ISSUE-109) regardless of
        # whether any frames were written; a zero-frame episode keeps the
        # empty trace.
        self._episode_traces.append(
            {"episode_index": summary.episode_idx, "trace_id": self._current_episode_trace}
        )

        if self._dataset is None:
            # No frames were written — nothing to save. Common during
            # smoke tests or zero-step episodes; not an error.
            self._n_episodes += 1
            if summary.success:
                self._n_success += 1
            return

        # episode_data carries per-episode metadata; lerobot v3 accepts
        # any extra keys here and lands them on meta/episodes/*. The
        # backfill of next.success across the episode's rows is
        # documented in the converter PR (PR4); for now the per-row
        # next.success is False on hardware and the episode-level
        # summary.success drives the aggregate dataset_success_rate.
        self._dataset.save_episode(parallel_encoding=True)
        self._n_episodes += 1
        if summary.success:
            self._n_success += 1
        _log.debug(
            "lerobot_save_episode",
            episode_idx=summary.episode_idx,
            success=summary.success,
            n_frames=summary.n_frames,
            cumulative_success=f"{self._n_success}/{self._n_episodes}",
        )

    def finalize(self) -> None:
        """Flush all pending encodes and write dataset-level metadata."""
        if self._finalized:
            return
        if self._dataset is not None:
            self._dataset.finalize()
            self._write_dataset_metadata()
        self._finalized = True

    # ── Internals ───────────────────────────────────────────────────────────

    def _create_dataset(self) -> Any:
        """Call ``LeRobotDataset.create`` from the pre-resolved feature_specs.

        ADR-0019: state, action, and per-camera shapes are all known at
        sink construction time — no first-frame derivation. This method
        just translates the FeatureSpec dict to the dict format
        ``lerobot.datasets.LeRobotDataset.create`` expects and opens the
        dataset on disk.
        """
        # lerobot import is deferred to here to keep the package
        # importable on hosts without lerobot installed.
        from lerobot.datasets import LeRobotDataset

        features: dict[str, Any] = {
            key: _featurespec_to_lerobot_dict(spec) for key, spec in self._feature_specs.items()
        }
        # ISSUE-109: declare the OTel correlation columns so add_frame
        # accepts them (lerobot rejects any frame key absent from the
        # features dict). They sit alongside the policy features as plain
        # string parquet columns.
        features.update({k: dict(v) for k, v in _TRACE_FEATURES.items()})

        # lerobot v3's fps is typed as int; round and warn if non-integer.
        fps_int = round(self._fps)
        if abs(fps_int - self._fps) > _FPS_INTEGER_TOLERANCE:
            _log.warning(
                "lerobot_fps_rounded",
                requested=self._fps,
                used=fps_int,
                hint="LeRobotDataset v3 only stores integer fps; non-integer fps would be lossy.",
            )

        # Ensure the parent exists but NOT the root itself —
        # LeRobotDatasetMetadata.create() calls root.mkdir(exist_ok=False)
        # and refuses to write into a pre-existing directory.
        self._root.parent.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset.create(
            repo_id=self._repo_id,
            fps=fps_int,
            features=features,
            root=self._root,
            robot_type=self._robot.name,
            use_videos=True,
            vcodec=self._vcodec,
        )
        _log.info(
            "lerobot_dataset_created",
            root=str(self._root),
            repo_id=self._repo_id,
            fps=fps_int,
            n_features=len(features),
            image_keys=list(self._image_keys),
            license=self._license,
            vcodec=self._vcodec,
        )
        return dataset

    def _write_dataset_metadata(self) -> None:
        """Append OpenRAL-specific metadata to ``meta/info.json``.

        v3's ``info.json`` carries a free-form ``metadata`` dict that we
        populate with the license, repo_id, dataset_success_rate, and the
        dataset-level OTel ``trace_ids`` (the distinct set of traces that
        produced these rows — ISSUE-109). The per-episode trace pointers
        go to the ``meta/openral_traces.json`` sidecar.
        """
        import json

        # Distinct, sorted, non-empty trace ids across all episodes.
        trace_ids = sorted({e["trace_id"] for e in self._episode_traces if e["trace_id"]})

        info_path = self._root / "meta" / "info.json"
        if not info_path.is_file():
            _log.warning("lerobot_info_json_missing", path=str(info_path))
            return
        info = json.loads(info_path.read_text())
        info.setdefault("metadata", {})
        info["metadata"]["license"] = self._license
        info["metadata"]["repo_id"] = self._repo_id
        info["metadata"]["robot_name"] = self._robot.name
        if self._n_episodes > 0:
            info["metadata"]["dataset_success_rate"] = self._n_success / self._n_episodes
            info["metadata"]["n_success_episodes"] = self._n_success
            info["metadata"]["n_episodes"] = self._n_episodes
        info["metadata"]["trace_ids"] = trace_ids
        info["metadata"]["n_traces"] = len(trace_ids)
        info_path.write_text(json.dumps(info, indent=2))

        # Episode-level sidecar: episode_index → producing trace_id. Kept
        # out of meta/episodes/*.parquet because v3 drops string features
        # from its per-episode stats; this small file is what lets
        # `openral replay` pivot at episode granularity without scanning
        # the (video-adjacent) data parquet.
        traces_path = self._root / "meta" / "openral_traces.json"
        traces_path.write_text(
            json.dumps({"dataset_trace_ids": trace_ids, "episodes": self._episode_traces}, indent=2)
        )
        _log.debug(
            "lerobot_info_metadata_updated",
            path=str(info_path),
            license=self._license,
            repo_id=self._repo_id,
            success_rate=info["metadata"].get("dataset_success_rate"),
            n_traces=len(trace_ids),
        )

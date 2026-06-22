"""Rosbag2ToLeRobotConverter — offline mcap rosbag2 → LeRobotDataset v3.

ADR-0019 PR4. Reads back the mcap file written by :class:`Rosbag2Sink`
(PR3) and produces a :class:`lerobot.datasets.LeRobotDataset` v3.0 via
the same :class:`LeRobotDatasetSink` the online sim path uses. Closes
the bidirectional bridge: hardware execution → mcap bag → on-disk
LeRobotDataset → HF Hub via `openral dataset push` (PR5).

The converter is deliberately narrow:

* Reads only the two openral topics (`/openral/tick`, `/openral/episode`).
  Camera streams on `/cameras/<id>/image_raw` are NOT joined in this PR;
  PR3 records zero-placeholder images in the bag (the hardware path's
  inline data isn't available without ROS), and PR2's `SensorRosPublisher`
  writes the real camera frames to ROS topics that this converter will
  read in a follow-up. The current converter therefore produces
  state-and-action-only datasets — every camera key in the robot's
  ``features_from_robot`` output gets a zero-shape video. That is
  honest: there are no real images in the bag to recover.

* Walks `/openral/episode` markers to segment the bag into discrete
  episodes. A bag with no Episode markers is rejected with a clean
  :class:`ROSConfigError` — the converter is not in the business of
  guessing where one episode ends and the next begins.

* Joins state + action at the recorded timestamps. mcap iter_messages
  is time-ordered; the converter zips Ticks with the preceding /
  trailing Episode markers into one episode per (PHASE_START,
  PHASE_END) pair.

Per CLAUDE.md §1.11 — exercised in `python/dataset/tests/test_converter.py`
against real bags written by the PR3 sink, with the resulting dataset
reloaded by a real :class:`lerobot.datasets.LeRobotDataset` reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import structlog
from openral_core.exceptions import ROSConfigError

from openral_dataset.bag import (
    PHASE_END,
    PHASE_START,
    TOPIC_EPISODE,
    TOPIC_IMAGE,
    TOPIC_TICK,
)

if TYPE_CHECKING:
    from openral_core import RobotDescription

__all__ = ["DatasetSummary", "Rosbag2ToLeRobotConverter"]

_log = structlog.get_logger(__name__)

_MCAP_INSTALL_HINT: Final[str] = (
    "mcap>=1.2 is required for Rosbag2ToLeRobotConverter. "
    "Install via `uv pip install 'openral-dataset[ros]'`."
)


@dataclass(frozen=True)
class DatasetSummary:
    """Bookkeeping result returned by :meth:`Rosbag2ToLeRobotConverter.from_bag`.

    Attributes:
        output_root: Path to the produced LeRobotDataset v3 root.
        n_episodes: Number of episodes converted.
        n_frames: Total frame count across all episodes.
        n_success: Number of episodes where PHASE_END.success was True.
        repo_id: The repo_id stored in ``meta/info.json``.
    """

    output_root: Path
    n_episodes: int
    n_frames: int
    n_success: int
    repo_id: str


@dataclass
class _EpisodeBuffer:
    """Per-episode accumulator used during the bag walk."""

    episode_idx: int
    task_string: str
    start_stamp_ns: int
    ticks: list[dict[str, object]]


class Rosbag2ToLeRobotConverter:
    """Offline replay of a `Rosbag2Sink`-produced mcap into a LeRobotDataset v3.

    Usage::

        summary = Rosbag2ToLeRobotConverter.from_bag(
            bag_path=Path("/tmp/run.mcap"),
            robot=robot,
            output_root=Path("/tmp/ds"),
        )

    The converter walks the mcap once, builds per-episode buffers from
    PHASE_START/PHASE_END pairs, then replays each episode through a
    :class:`LeRobotDatasetSink` so the on-disk format is identical to
    what the online sim path produces. ``next.success`` for every frame
    in an episode is set from the episode's PHASE_END marker
    (episode-level success, broadcast per-frame; PR0/ADR-0019 §3
    discusses why the per-frame field is uniform across the episode).
    """

    @classmethod
    def from_bag(
        cls,
        *,
        bag_path: Path | str,
        robot: RobotDescription,
        output_root: Path | str,
        repo_id: str | None = None,
        license: str = "CC-BY-4.0",
        fps: float | None = None,
    ) -> DatasetSummary:
        """Convert an mcap bag to a LeRobotDataset v3 dataset.

        Args:
            bag_path: Path to the input ``.mcap`` file produced by
                :class:`Rosbag2Sink`.
            robot: Robot description used to bind feature shapes (state
                vector, action dim). Must match what the recorder used
                at write time — mismatched shapes are caught at
                ``add_frame`` time by lerobot.
            output_root: Destination LeRobotDataset root directory.
                Must NOT pre-exist; lerobot refuses to overwrite.
            repo_id: Override the resulting dataset's repo_id. Defaults
                to ``openral/dataset-<robot.name>``.
            license: SPDX license string carried in
                ``meta/info.json["metadata"]["license"]``.
            fps: Frames-per-second to record in the dataset. Defaults
                to ``robot.action_spec.control_freq_hz`` or 30.0.

        Returns:
            :class:`DatasetSummary` describing what was written.

        Raises:
            ROSConfigError: When mcap is unimportable, the bag is
                missing, has no `/openral/episode` markers, or has
                malformed Episode messages.
        """
        try:
            import mcap  # noqa: F401  # reason: presence probe
        except ImportError as exc:
            raise ROSConfigError(_MCAP_INSTALL_HINT) from exc

        bag = Path(bag_path)
        if not bag.is_file():
            raise ROSConfigError(f"Rosbag2ToLeRobotConverter: bag_path {bag} does not exist")

        # Resolve fps once so both the recorder and the sink land on
        # the same value.
        if fps is None:
            action_spec = robot.action_spec
            resolved_fps = (
                float(action_spec.control_freq_hz)
                if action_spec is not None and action_spec.control_freq_hz
                else 30.0
            )
        else:
            resolved_fps = float(fps)

        episodes, images_by_step = cls._walk_bag(bag)
        if not episodes:
            raise ROSConfigError(
                f"Rosbag2ToLeRobotConverter: bag {bag} has no /openral/episode markers; "
                "was it recorded by openral_dataset.Rosbag2Sink?"
            )

        # Local import keeps the converter importable when lerobot is
        # absent — same pattern as LeRobotDatasetSink itself.
        from openral_dataset import LeRobotDatasetSink, RolloutRecorder

        resolved_repo_id = repo_id if repo_id is not None else f"openral/dataset-{robot.name}"
        # Derive feature shapes from the recorded bag itself — the inline
        # arrays + image frames are authoritative. This makes from-bag work
        # for ANY robot, including ones whose proprio/action layout lives on
        # the active rSkill's contracts rather than RobotDescription's
        # observation_spec/action_spec (e.g. franka_panda). Falls back to the
        # robot spec for legacy metadata-only bags (overrides stay None).
        state_override, action_override, camera_override = cls._shape_overrides_from_bag(
            episodes, images_by_step
        )
        sink = LeRobotDatasetSink(
            root=output_root,
            robot=robot,
            fps=resolved_fps,
            repo_id=resolved_repo_id,
            license=license,
            state_shape=state_override,
            action_dim=action_override,
            camera_shape=camera_override,
        )
        # The recorder is the canonical way to drive a DatasetSink; reuse
        # it here instead of inlining the sink's lifecycle, so any
        # future helper that lands on the recorder (e.g. dataset_repo_id
        # OTel attribute) also runs during conversion.
        rec = RolloutRecorder(
            robot=robot,
            task_string=episodes[0].task_string,
            fps=resolved_fps,
            sinks=[sink],
            repo_id=resolved_repo_id,
        )

        n_frames = 0
        n_success = 0
        for ep_buf in episodes:
            rec.episode_start(task_string=ep_buf.task_string)
            for tick in ep_buf.ticks:
                cls._replay_tick(rec, tick, robot, images_by_step)
                n_frames += 1
            success = ep_buf.ticks[-1].get("_episode_success", False) if ep_buf.ticks else False
            rec.episode_end(success=bool(success))
            if success:
                n_success += 1
        rec.finalize()

        summary = DatasetSummary(
            output_root=Path(output_root),
            n_episodes=len(episodes),
            n_frames=n_frames,
            n_success=n_success,
            repo_id=resolved_repo_id,
        )
        _log.info(
            "rosbag2_to_lerobot.converted",
            bag=str(bag),
            output_root=str(output_root),
            n_episodes=summary.n_episodes,
            n_frames=summary.n_frames,
            n_success=summary.n_success,
            repo_id=resolved_repo_id,
        )
        return summary

    # ── Internals ───────────────────────────────────────────────────────────

    @classmethod
    def _walk_bag(
        cls, bag_path: Path
    ) -> tuple[list[_EpisodeBuffer], dict[tuple[int, int], dict[str, Any]]]:
        """First pass: walk the mcap and group Ticks under their Episode markers.

        Returns ``(episodes, images_by_step)`` where ``episodes`` is one
        `_EpisodeBuffer` per matched (PHASE_START, PHASE_END) pair and
        ``images_by_step`` maps ``(episode_idx, step_idx)`` to a
        ``{camera: HxWxC uint8 array}`` dict decoded from the bag's
        ``/openral/dataset/image`` messages. Ticks outside any episode are
        silently dropped (rare; only happens if a bag was truncated
        mid-episode). A PHASE_START with no matching PHASE_END is
        treated as a failure episode (the deactivate-with-open-episode
        contract).
        """
        from mcap.reader import make_reader

        episodes: list[_EpisodeBuffer] = []
        images_by_step: dict[tuple[int, int], dict[str, Any]] = {}
        current: _EpisodeBuffer | None = None
        with bag_path.open("rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages():
                payload = json.loads(message.data.decode("utf-8"))
                if channel.topic == TOPIC_IMAGE:
                    key = (int(payload["episode_idx"]), int(payload["step_idx"]))
                    images_by_step.setdefault(key, {})[str(payload["camera"])] = cls._decode_image(
                        payload
                    )
                    continue
                if channel.topic == TOPIC_EPISODE:
                    phase = payload.get("phase")
                    if phase == PHASE_START:
                        if current is not None:
                            # PHASE_START without an intervening
                            # PHASE_END — treat the previous episode
                            # as a failure (no marker to read from).
                            cls._mark_episode_success(current, success=False)
                            episodes.append(current)
                        current = _EpisodeBuffer(
                            episode_idx=int(payload.get("episode_idx", len(episodes))),
                            task_string=str(payload.get("task_string", "")),
                            start_stamp_ns=int(payload.get("stamp_ns", 0)),
                            ticks=[],
                        )
                    elif phase == PHASE_END:
                        if current is None:
                            # Stray PHASE_END with no PHASE_START —
                            # ignore; the bag is malformed but a
                            # missing PHASE_START isn't worth aborting.
                            continue
                        cls._mark_episode_success(
                            current, success=bool(payload.get("success", False))
                        )
                        episodes.append(current)
                        current = None
                    else:
                        raise ROSConfigError(
                            f"Rosbag2ToLeRobotConverter: unknown Episode phase "
                            f"{phase!r}; expected PHASE_START=0 or PHASE_END=1"
                        )
                elif channel.topic == TOPIC_TICK and current is not None:
                    current.ticks.append(payload)
                # Other topics (e.g. PR2's camera streams) are
                # ignored for now; PR4-follow-up wires them in.
        if current is not None:
            # Open episode at EOF — treat as failure.
            cls._mark_episode_success(current, success=False)
            episodes.append(current)
        return episodes, images_by_step

    @staticmethod
    def _shape_overrides_from_bag(
        episodes: list[_EpisodeBuffer],
        images_by_step: dict[tuple[int, int], dict[str, Any]],
    ) -> tuple[tuple[int, ...] | None, int | None, tuple[int, int] | None]:
        """Derive (state_shape, action_dim, camera_shape) from recorded data.

        Returns ``None`` for any shape the bag does not carry inline (legacy
        metadata-only bags), so the sink falls back to the robot spec.
        """
        first_tick = next((t for ep in episodes for t in ep.ticks), None)
        state_override: tuple[int, ...] | None = None
        action_override: int | None = None
        if first_tick is not None:
            state_raw = first_tick.get("observation_state")
            if isinstance(state_raw, list) and state_raw:
                state_override = (len(state_raw),)
            action_raw = first_tick.get("action")
            if isinstance(action_raw, list) and action_raw:
                action_override = len(action_raw)
        camera_override: tuple[int, int] | None = None
        first_images = next(iter(images_by_step.values()), None)
        if first_images:
            any_arr = next(iter(first_images.values()))
            camera_override = (int(any_arr.shape[0]), int(any_arr.shape[1]))
        return state_override, action_override, camera_override

    @staticmethod
    def _decode_image(payload: dict[str, Any]) -> Any:
        """Decode a ``/openral/dataset/image`` payload to a HxWxC uint8 array."""
        import base64

        raw = base64.b64decode(str(payload["data_b64"]))
        height = int(payload["height"])
        width = int(payload["width"])
        channels = int(payload["channels"])
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)

    @staticmethod
    def _mark_episode_success(buf: _EpisodeBuffer, *, success: bool) -> None:
        """Stamp the episode-level success flag onto every tick.

        Stored as a private ``_episode_success`` key on each tick dict
        so :meth:`_replay_tick` can read it back without consulting the
        surrounding episode buffer.
        """
        for tick in buf.ticks:
            tick["_episode_success"] = success

    @staticmethod
    def _replay_tick(
        rec: object,
        tick: dict[str, object],
        robot: RobotDescription,
        images_by_step: dict[tuple[int, int], dict[str, Any]],
    ) -> None:
        """Drive `RolloutRecorder.record_frame` from a Tick payload.

        ``observation_state`` / ``action`` are read from the inline
        arrays the enriched :class:`Rosbag2Sink` now writes; per-camera
        pixels are joined from the ``/openral/dataset/image`` messages
        keyed by ``(episode_idx, step_idx)``. For legacy metadata-only
        bags (no inline arrays / images) the converter falls back to a
        zero vector / zero frame of the robot's declared shape, so old
        bags still convert. reward / terminated / truncated round-trip
        from the Tick payload verbatim.
        """
        # ADR-0019: shapes MUST match the sink's declared feature shapes
        # (state from RobotDescription / rSkill manifest contract;
        # cameras from SensorSpec.intrinsics). The sink's strict per-frame
        # shape validation catches any drift between the bag records and
        # the declared contract.
        obs_spec = robot.observation_spec
        state_shape = tuple(obs_spec.state_shape) if obs_spec is not None else (1,)
        action_dim = robot.action_spec.dim if robot.action_spec is not None else 1

        # Prefer the bag's inline arrays at their NATURAL recorded length —
        # the sink's feature schema was derived from the same data
        # (_shape_overrides_from_bag), so this matches even for robots with no
        # observation_spec. Legacy metadata-only bags fall back to a zero
        # vector of the robot's declared shape.
        state_raw = tick.get("observation_state")
        if isinstance(state_raw, list) and state_raw:
            state = np.asarray(state_raw, dtype=np.float32)
        else:
            state = np.zeros(state_shape, dtype=np.float32)
        action_raw = tick.get("action")
        if isinstance(action_raw, list) and action_raw:
            action = np.asarray(action_raw, dtype=np.float32)
        else:
            action = np.zeros(action_dim, dtype=np.float32)

        # Real per-camera frames recorded for this step, if present.
        episode_idx = int(cast("float | int", tick.get("episode_idx", 0)))
        step_idx = int(cast("float | int", tick.get("step_idx", 0)))
        recorded_images = images_by_step.get((episode_idx, step_idx), {})
        images: dict[str, np.ndarray[Any, Any]] = {}
        for sensor in robot.sensors:
            if sensor.vla_feature_key is None or sensor.intrinsics is None:
                continue
            stripped = sensor.vla_feature_key.removeprefix("observation.images.")
            channels = 1 if sensor.modality in {"depth", "ir", "thermal"} else 3
            if stripped in recorded_images:
                images[stripped] = recorded_images[stripped]
            else:
                # Legacy bag without inline pixels — zero frame at the
                # declared intrinsic resolution keeps the contract valid.
                images[stripped] = np.zeros(
                    (int(sensor.intrinsics.height), int(sensor.intrinsics.width), channels),
                    dtype=np.uint8,
                )

        # JSON decode produces `object`-typed values for dict[str, object]
        # entries; cast to concrete numerics for mypy strict-mode.
        reward_raw = tick.get("reward", 0.0) or 0.0
        stamp_raw = tick.get("stamp_ns", 0)
        # ISSUE-109: replay the bag's ORIGINAL (trace_id, span_id) rather
        # than letting record_frame capture this conversion process's own
        # span context — the on-disk frame must point back at the rollout
        # that produced it, not the offline convert run.
        rec.record_frame(  # type: ignore[attr-defined]
            observation_state=state,
            images=images,
            action=action,
            reward=float(cast("float | int", reward_raw)),
            terminated=bool(tick.get("terminated", False)),
            truncated=bool(tick.get("truncated", False)),
            stamp_ns=int(cast("float | int", stamp_raw)),
            trace_id=str(tick.get("trace_id", "") or ""),
            span_id=str(tick.get("span_id", "") or ""),
        )

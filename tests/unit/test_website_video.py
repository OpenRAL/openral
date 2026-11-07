"""Unit tests for the clean website world-view MP4 writer.

Exercises ``openral_sim._website_video.save_world_mp4`` and the
``videos.json`` manifest merge with a real ``EpisodeResult`` and real
imageio/ffmpeg muxing — no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_sim._website_video import (
    append_video_manifest,
    save_world_mp4,
    write_world_grid,
    write_world_videos,
)
from openral_sim.rollout import EpisodeResult


def _result_with_frames(n: int, h: int, w: int, *, success: bool) -> EpisodeResult:
    """A real EpisodeResult carrying ``n`` non-square HWC uint8 frames."""
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]
    return EpisodeResult(success=success, steps=n, frames=frames)


def test_save_world_mp4_writes_square_video(tmp_path: Path) -> None:
    import imageio.v2 as iio

    out = save_world_mp4(
        _result_with_frames(8, h=200, w=320, success=True),
        tmp_path / "scene_rskill_success.mp4",
        size=256,
        square=True,
    )
    assert out.exists()
    reader = iio.get_reader(out, format="ffmpeg")  # type: ignore[arg-type]  # reason: imageio stub types format as Format, str is the documented form
    try:
        frame = reader.get_data(0)  # first frame
    finally:
        reader.close()
    assert frame.shape[0] == 256 and frame.shape[1] == 256


def _read_all(path: Path) -> list[NDArray[np.uint8]]:
    import imageio.v2 as iio

    reader = iio.get_reader(path, format="ffmpeg")  # type: ignore[arg-type]  # reason: imageio stub types format as Format, str is the documented form
    try:
        n = reader.count_frames()  # type: ignore[attr-defined]  # reason: ffmpeg plugin Reader method, absent from the base stub
        return [np.asarray(reader.get_data(i), dtype=np.uint8) for i in range(n)]
    finally:
        reader.close()


def test_save_world_mp4_keeps_native_aspect_by_default(tmp_path: Path) -> None:
    out = save_world_mp4(_result_with_frames(4, h=240, w=320, success=True), tmp_path / "a.mp4")
    assert _read_all(out)[0].shape == (768, 1024, 3)  # 4:3 kept, long edge = size


def _near(frame: NDArray[np.uint8], colour: tuple[int, int, int]) -> bool:
    """The outline's left edge (mid-height, its 2 px) is ``colour`` (yuv420p chroma blur)."""
    h = frame.shape[0]
    edge = frame[h // 4 : 3 * h // 4, 0:2].reshape(-1, 3).mean(0)
    return bool(np.abs(edge - colour).max() < 25)


def test_world_border_is_grey_while_running_then_the_grade_colour(tmp_path: Path) -> None:
    """A real graded episode: ungraded, then partial from step 3, ended partial (reward 0.5)."""
    grey = np.full((96, 128, 3), 30, np.uint8)
    r = EpisodeResult(
        success=False,
        steps=5,
        total_reward=0.5,
        frames=[grey] * 5,
        frame_grades=[None, ("fail", 0.0), ("fail", 0.0), ("partial", 0.5), ("partial", 0.5)],
        final_info={"placement_grade": "partial", "placement_score": 0.5},
    )
    frames = _read_all(save_world_mp4(r, tmp_path / "p.mp4", fps=4, size=128, min_duration_s=0))
    assert len(frames) == 5 + 4  # + a 1 s hold of the outcome
    assert all(_near(f, (128, 128, 128)) for f in frames[:3])
    assert all(_near(f, (240, 210, 30)) for f in frames[3:])  # partial = yellow
    # an ungraded fail: grey until the last frame, then red
    fail = EpisodeResult(success=False, steps=3, frames=[grey] * 3)
    frames = _read_all(save_world_mp4(fail, tmp_path / "f.mp4", fps=4, size=128))
    assert _near(frames[1], (128, 128, 128)) and _near(frames[-1], (220, 40, 40))


def test_save_world_mp4_empty_frames_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no world frames"):
        save_world_mp4(EpisodeResult(success=False), tmp_path / "x.mp4")


def test_save_world_mp4_rejects_non_mp4_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only writes MP4"):
        save_world_mp4(_result_with_frames(2, 64, 64, success=True), tmp_path / "x.gif")


def test_save_world_mp4_rejects_nonpositive_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        save_world_mp4(_result_with_frames(2, 64, 64, success=True), tmp_path / "x.mp4", size=0)


def test_write_world_videos_names_and_manifests(tmp_path: Path) -> None:
    episodes = [
        _result_with_frames(4, 96, 128, success=True),
        _result_with_frames(4, 96, 128, success=False),
    ]
    records = write_world_videos(
        episodes,
        tmp_path,
        scene="libero_spatial",
        rskill="smolvla-libero",
        section="benchmark",
        size=128,
    )
    # Multi-episode → _ep<i> + per-episode success/fail suffix.
    assert (tmp_path / "libero_spatial_smolvla-libero_ep0_success.mp4").exists()
    assert (tmp_path / "libero_spatial_smolvla-libero_ep1_fail.mp4").exists()
    manifest = json.loads((tmp_path / "videos.json").read_text())
    assert len(manifest) == len(records) == 2
    assert {r["section"] for r in manifest} == {"benchmark"}
    assert [r["success"] for r in manifest] == [True, False]


def test_write_world_grid_tiles_every_episode_keeping_aspect(tmp_path: Path) -> None:
    import imageio.v2 as iio

    episodes = [
        _result_with_frames(6, 90, 160, success=True),
        _result_with_frames(3, 90, 160, success=False),
        _result_with_frames(4, 90, 160, success=False),
    ]
    episodes[2].total_reward = 0.25
    out = write_world_grid(episodes, tmp_path / "grid.mp4", tile_height=90, fps=10)
    reader = iio.get_reader(out, format="ffmpeg")  # type: ignore[arg-type]  # reason: imageio stub types format as Format, str is the documented form
    try:
        n_frames, first = reader.count_frames(), reader.get_data(0)  # type: ignore[attr-defined]  # reason: ffmpeg plugin Reader method, absent from the base stub
    finally:
        reader.close()
    # 3 clips -> 2 columns x 2 rows of 160 x 90 tiles; the shortest clips hold their
    # last frame until the longest (6 frames) ends, plus a 1 s (10 frame) outcome hold.
    assert n_frames == 6 + 10
    assert first.shape == (180, 320, 3)
    assert write_world_grid(
        [_result_with_frames(2, 64, 64, success=True)], tmp_path / "one.mp4"
    ).exists()
    with pytest.raises(ValueError, match="at least one episode"):
        write_world_grid([EpisodeResult(success=False)], tmp_path / "none.mp4")


def test_write_world_grid_rounds_an_odd_tile_height_to_even(tmp_path: Path) -> None:
    """yuv420p cannot encode odd sides; a single 91 px tile would otherwise fail in ffmpeg."""
    import imageio.v2 as iio

    out = write_world_grid(
        [_result_with_frames(2, 64, 96, success=True)], tmp_path / "odd.mp4", tile_height=91
    )
    reader = iio.get_reader(out, format="ffmpeg")  # type: ignore[arg-type]  # reason: imageio stub types format as Format, str is the documented form
    try:
        assert reader.get_data(0).shape[0] == 90
    finally:
        reader.close()
    with pytest.raises(ValueError, match="at least 2"):
        write_world_grid(
            [_result_with_frames(2, 64, 96, success=True)], tmp_path / "x.mp4", tile_height=1
        )


def test_write_world_videos_can_add_the_grid(tmp_path: Path) -> None:
    write_world_videos(
        [
            _result_with_frames(3, 64, 96, success=True),
            _result_with_frames(2, 64, 96, success=False),
        ],
        tmp_path,
        scene="pusht",
        rskill="diffusion-pusht",
        section="sim",
        size=64,
        grid=True,
    )
    assert (tmp_path / "pusht_diffusion-pusht_grid.mp4").exists()
    manifest = json.loads((tmp_path / "videos.json").read_text())
    assert {r["file"] for r in manifest} == {
        "pusht_diffusion-pusht_ep0_success.mp4",
        "pusht_diffusion-pusht_ep1_fail.mp4",
    }


def test_write_world_videos_single_episode_omits_ep_index(tmp_path: Path) -> None:
    records = write_world_videos(
        [_result_with_frames(3, 64, 64, success=True)],
        tmp_path,
        scene="pusht",
        rskill="diffusion-pusht",
        section="benchmark",
        size=64,
    )
    assert (tmp_path / "pusht_diffusion-pusht_success.mp4").exists()
    assert len(records) == 1


def test_write_world_videos_flattens_slashed_scene_id(tmp_path: Path) -> None:
    # Robocasa/SimplerEnv scene ids carry slashes; the filename must not nest.
    write_world_videos(
        [_result_with_frames(2, 64, 64, success=False)],
        tmp_path,
        scene="robocasa/gr1/PnPCupToDrawerClose",
        rskill="rldx1-ft-gr1-nf4",
        section="benchmark",
        size=64,
    )
    expected = tmp_path / "robocasa_gr1_PnPCupToDrawerClose_rldx1-ft-gr1-nf4_fail.mp4"
    assert expected.exists()
    # No nested subdirectories were created.
    assert not (tmp_path / "robocasa").exists()
    # Manifest keeps the original (slashed) scene id for display.
    rec = json.loads((tmp_path / "videos.json").read_text())[0]
    assert rec["scene"] == "robocasa/gr1/PnPCupToDrawerClose"
    assert rec["file"] == expected.name


def test_manifest_merges_and_replaces_by_file(tmp_path: Path) -> None:
    manifest = tmp_path / "videos.json"
    append_video_manifest(
        manifest,
        [{"file": "a_success.mp4", "scene": "a", "success": True}],
    )
    # Append a new entry, and re-record "a" as a fail — same file is replaced,
    # not duplicated.
    append_video_manifest(
        manifest,
        [
            {"file": "b_fail.mp4", "scene": "b", "success": False},
            {"file": "a_success.mp4", "scene": "a", "success": False},
        ],
    )
    records = json.loads(manifest.read_text())
    by_file = {r["file"]: r for r in records}
    assert set(by_file) == {"a_success.mp4", "b_fail.mp4"}
    assert by_file["a_success.mp4"]["success"] is False  # replaced, not duplicated
    assert len(records) == 2


def test_manifest_noop_on_empty_records(tmp_path: Path) -> None:
    manifest = tmp_path / "videos.json"
    append_video_manifest(manifest, [])
    assert not manifest.exists()


def test_sim_run_videos_play_at_the_robot_control_rate() -> None:
    """One frame per env step: sim-run videos are written at the robot's control rate
    (OpenArm 30 Hz), not the writers' 20 fps default, so they play in real time."""
    from openral_core import RobotDescription
    from openral_sim.cli import _control_rate_hz, _robot_control_rate_hz

    openarm = RobotDescription.from_yaml(
        str(Path(__file__).parents[2] / "robots/openarm/robot.yaml")
    )
    assert openarm.action_spec is not None
    assert openarm.action_spec.control_freq_hz is not None
    assert _robot_control_rate_hz("openarm") == float(openarm.action_spec.control_freq_hz) == 30.0
    assert _control_rate_hz(openarm.model_copy(update={"action_spec": None})) == 30.0
    assert _robot_control_rate_hz(None) == 30.0

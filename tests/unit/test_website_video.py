"""Unit tests for the clean website world-view MP4 writer.

Exercises ``openral_sim._website_video.save_world_mp4`` and the
``videos.json`` manifest merge with a real :class:`EpisodeResult` and real
imageio/ffmpeg muxing — no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from openral_sim._website_video import (
    append_video_manifest,
    save_world_mp4,
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
    )
    assert out.exists()
    reader = iio.get_reader(out, format="ffmpeg")
    try:
        frame = reader.get_data(0)  # first frame
    finally:
        reader.close()
    assert frame.shape[0] == 256 and frame.shape[1] == 256


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

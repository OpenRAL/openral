"""Clean single-view MP4 helper for website hero videos.

Unlike ``openral_sim._video`` (the two-band *debug* montage), this writes
*only* the world/viewer render — the frame an operator sees in the MuJoCo
viewer — with no VLA-input panel, no joint plot, and no burned-in overlays.

The website component (``website/``) is responsible for everything visual:
square crop (``object-fit: cover``), rounded corners, the benchmark/rSkill
labels, and the SUCCESS/FAIL status badge are all styled DOM elements layered
over the ``<video>``. This keeps text crisp and themeable and lets the same
clip be restyled without re-encoding.

The output square edge (``size``) controls only the written canvas; source
sharpness is bounded by the scene's native render resolution. We never touch
the policy's observation resolution, so recording does not change task outcomes.

Implementation notes
--------------------
* Uses imageio + imageio-ffmpeg for MP4 muxing (libx264, yuv420p) — same codec
  path as ``openral_sim._video``.
* Each frame is center-cropped to a square, then resized to ``size × size`` so
  the file is display-ready square (the component still cover-crops, harmlessly).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from openral_sim.rollout import EpisodeResult

_RGB_CHANNELS = 3
_GRAYSCALE_NDIM = 2
_DEFAULT_SIZE = 1024
#: Tile height of the ``--video-grid`` composite; 4 x 4 tiles of 448 px fit a 1792 px frame.
_GRID_TILE_HEIGHT = 448
#: yuv420p subsamples chroma 2x2, so every encoded side must be even (>= 2).
_MIN_EVEN_SIDE = 2


def save_world_mp4(
    result: EpisodeResult,
    path: Path,
    *,
    fps: int = 20,
    size: int = _DEFAULT_SIZE,
    min_duration_s: float = 2.0,
) -> Path:
    """Write a clean square world-view MP4 for one ``EpisodeResult``.

    Args:
        result: An ``EpisodeResult`` produced by ``SimRunner`` with
            ``record_video=True``. Only ``frames`` (the rollout/world view) is
            used; ``vla_input_frames`` and ``joint_positions`` are ignored.
        path: Destination ``.mp4`` file. Parent directories are created.
        fps: Playback frame rate.
        size: Output square edge in pixels. Each frame is center-cropped to a
            square and resized to ``size × size``.
        min_duration_s: Minimum playback length. Short successful rollouts
            hold the final frame so website clips remain watchable.

    Returns:
        The output path.

    Raises:
        ValueError: If ``result.frames`` is empty (set ``record_video=True``
            first), if ``path`` does not end in ``.mp4`` (this helper is
            ffmpeg/x264/yuv420p only), or if ``size`` is not positive.

    Example:
        >>> # given an EpisodeResult `r` captured with record_video=True
        >>> from pathlib import Path
        >>> save_world_mp4(r, Path("libero_spatial_smolvla-libero_success.mp4"))  # doctest: +SKIP
        PosixPath('libero_spatial_smolvla-libero_success.mp4')
    """
    if not result.frames:
        raise ValueError(
            "EpisodeResult has no world frames; set SimEnvironment.record_video=True "
            "before driving SimRunner to capture rollout frames"
        )
    if path.suffix.lower() != ".mp4":
        raise ValueError(
            f"save_world_mp4 only writes MP4 (libx264/yuv420p); got path={path!s} "
            f"with suffix {path.suffix!r}. Use a '.mp4' extension."
        )
    if size <= 0:
        raise ValueError(f"size must be positive; got {size}")
    if min_duration_s < 0.0:
        raise ValueError(f"min_duration_s must be non-negative; got {min_duration_s}")

    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as iio

    writer = iio.get_writer(
        str(path),
        format="ffmpeg",  # type: ignore[arg-type]  # reason: imageio v2 stubs type format as Format enum but accept plugin strings at runtime
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,  # no resize; we build exact-size canvases
        pixelformat="yuv420p",
    )
    try:
        frames = list(result.frames)
        min_frames = int(np.ceil(float(fps) * min_duration_s))
        if frames and len(frames) < min_frames:
            frames.extend([frames[-1]] * (min_frames - len(frames)))
        for frame in frames:
            writer.append_data(_square(frame, size))
    finally:
        writer.close()
    return path


def write_world_grid(
    episodes: list[EpisodeResult],
    path: Path,
    *,
    tile_height: int = _GRID_TILE_HEIGHT,
    fps: int = 20,
) -> Path:
    """Write every episode's world render as one tiled MP4 (``ceil(sqrt(n))`` columns).

    Each tile keeps its frame's aspect at ``tile_height`` px (no crop, so nothing at
    the frame's edges is lost), carries an ``ep<i>`` label and its outcome, and holds
    its last frame until the longest episode ends. The outline is green for a
    successful episode, amber for a reward above zero, red otherwise -- the grade a
    graded scene (``placement_success``) scored, read from ``EpisodeResult``.

    Args:
        episodes: The rollout results; episodes without ``frames`` are skipped.
        path: Output ``.mp4``.
        tile_height: Height of one tile in pixels (rounded down to even, as yuv420p
            needs); the width follows the frame's aspect.
        fps: Playback frame rate.

    Raises:
        ValueError: No episode carries frames, ``path`` is not an ``.mp4``, or
            ``tile_height`` is below 2.

    Example:
        >>> write_world_grid(results, Path("run_grid.mp4"), tile_height=360)  # doctest: +SKIP
    """
    from PIL import Image, ImageDraw

    clips = [(i, r) for i, r in enumerate(episodes) if r.frames]
    if not clips:
        raise ValueError("write_world_grid needs at least one episode with world frames")
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"write_world_grid only writes MP4; got path={path!s}")
    if tile_height < _MIN_EVEN_SIDE:
        raise ValueError(f"tile_height must be at least {_MIN_EVEN_SIDE}; got {tile_height}")
    tile_height -= (
        tile_height % 2
    )  # yuv420p (and imageio, with macro_block_size=1) needs even sides
    h0, w0 = clips[0][1].frames[0].shape[:2]
    tile_w = round(w0 * tile_height / h0 / 2) * 2
    cols = int(np.ceil(np.sqrt(len(clips))))
    rows = int(np.ceil(len(clips) / cols))
    longest = max(len(r.frames) for _, r in clips)
    border = max(2, tile_height // 60)

    def tile(frame: NDArray[np.uint8], i: int, r: EpisodeResult) -> NDArray[np.uint8]:
        img = Image.fromarray(_rgb(frame)).resize((tile_w, tile_height), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(img)
        colour = (
            (40, 200, 60) if r.success else (235, 170, 20) if r.total_reward > 0 else (220, 40, 40)
        )
        draw.rectangle([0, 0, tile_w - 1, tile_height - 1], outline=colour, width=border)
        text = f"ep{i}  " + ("success" if r.success else f"reward {r.total_reward:.2f}")
        draw.text((border + 4, border + 2), text, fill=(255, 255, 255))
        return np.asarray(img, dtype=np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as iio

    writer = iio.get_writer(
        str(path),
        format="ffmpeg",  # type: ignore[arg-type]  # reason: imageio v2 stubs type format as Format enum but accept plugin strings at runtime
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pixelformat="yuv420p",
    )
    try:
        for t in range(longest):
            canvas = np.zeros((rows * tile_height, cols * tile_w, _RGB_CHANNELS), dtype=np.uint8)
            for k, (i, r) in enumerate(clips):
                frame = r.frames[min(t, len(r.frames) - 1)]
                y, x = (k // cols) * tile_height, (k % cols) * tile_w
                canvas[y : y + tile_height, x : x + tile_w] = tile(frame, i, r)
            writer.append_data(canvas)
    finally:
        writer.close()
    return path


def write_world_videos(
    episodes: list[EpisodeResult],
    out_dir: Path,
    *,
    scene: str,
    rskill: str,
    section: str,
    size: int = _DEFAULT_SIZE,
    fps: int = 20,
    grid: bool = False,
) -> list[dict[str, Any]]:
    """Write one clean world MP4 per episode + update ``out_dir/videos.json``.

    Shared by ``openral sim run --video-style world`` and ``openral benchmark
    scene --save-video``. Files are named
    ``<scene>_<rskill>_<success|fail>.mp4`` (``_ep<i>`` inserted for
    multi-episode runs).

    Args:
        episodes: The rollout results; only those with captured ``frames`` are
            written (set ``record_video=True`` on the run).
        out_dir: Destination directory (created if absent). Holds the MP4s and
            the merged ``videos.json`` manifest.
        scene: Scene id for the filename + manifest (e.g. ``libero_spatial``).
        rskill: rSkill basename for the filename + manifest (e.g. ``smolvla-libero``).
        section: Website section the clip belongs to (``benchmark`` / ``sim`` / ``deploy``).
        size: Square output edge in pixels, passed to ``save_world_mp4``.
        fps: Playback frame rate.
        grid: Also write ``<scene>_<rskill>_grid.mp4``, every episode tiled in one
            clip (``write_world_grid``); not a website clip, so not in the manifest.

    Returns:
        The manifest records appended this call (one per written video).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Scene/task ids carry slashes (e.g. "robocasa/gr1/PnPCupToDrawerClose");
    # flatten to a single filename token so nothing nests into subdirs.
    scene_slug = _slug(scene)
    rskill_slug = _slug(rskill)
    multi = len(episodes) > 1
    records: list[dict[str, Any]] = []
    for i, r in enumerate(episodes):
        if not r.frames:
            print(f"  ep{i}: no world frames captured (record_video off?)")
            continue
        status = "success" if r.success else "fail"
        ep = f"_ep{i}" if multi else ""
        path = out_dir / f"{scene_slug}_{rskill_slug}{ep}_{status}.mp4"
        # The filename embeds the outcome, so a re-run whose outcome flipped
        # would otherwise leave the previous run's opposite-status MP4 (and
        # its manifest record) contradicting the new one. Remove the stale
        # sibling before writing.
        other_status = "fail" if r.success else "success"
        stale = out_dir / f"{scene_slug}_{rskill_slug}{ep}_{other_status}.mp4"
        if stale.exists():
            stale.unlink()
            _drop_manifest_records(out_dir / "videos.json", {stale.name})
            print(f"  removed stale {stale}")
        out = save_world_mp4(r, path, size=size, fps=fps)
        print(f"  wrote {out}")
        records.append(
            {
                "section": section,
                "scene": scene,
                "rskill": rskill,
                "success": bool(r.success),
                "file": out.name,
                "steps": r.steps,
                "fps": fps,
            }
        )
    append_video_manifest(out_dir / "videos.json", records)
    if grid and any(r.frames for r in episodes):
        out = write_world_grid(episodes, out_dir / f"{scene_slug}_{rskill_slug}_grid.mp4", fps=fps)
        print(f"  wrote {out}")
    return records


def append_video_manifest(manifest: Path, records: list[dict[str, Any]]) -> None:
    """Merge ``records`` into ``videos.json``, replacing same-``file`` entries.

    No-op when ``records`` is empty (a run that captured no frames must not
    truncate or rewrite an existing manifest).
    """
    if not records:
        return
    existing: list[dict[str, Any]] = []
    if manifest.exists():
        loaded = json.loads(manifest.read_text())
        if isinstance(loaded, list):
            existing = loaded
    new_files = {rec["file"] for rec in records}
    merged = [rec for rec in existing if rec.get("file") not in new_files] + records
    manifest.write_text(json.dumps(merged, indent=2))
    print(f"  updated {manifest}")


def _drop_manifest_records(manifest: Path, file_names: set[str]) -> None:
    """Remove ``videos.json`` records whose ``file`` was deleted from disk."""
    if not manifest.exists():
        return
    loaded = json.loads(manifest.read_text())
    if not isinstance(loaded, list):
        return
    kept = [rec for rec in loaded if rec.get("file") not in file_names]
    if len(kept) != len(loaded):
        manifest.write_text(json.dumps(kept, indent=2))


def _slug(s: str) -> str:
    """Flatten an id into a single filename token (no path separators)."""
    return s.replace("/", "_").replace("\\", "_")


def _rgb(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """``frame`` as HWC RGB (a grayscale or single-channel frame is replicated)."""
    if frame.ndim == _GRAYSCALE_NDIM:
        return np.stack([frame] * _RGB_CHANNELS, axis=-1)
    if frame.shape[2] != _RGB_CHANNELS:
        return np.repeat(frame[:, :, :1], _RGB_CHANNELS, axis=2)
    return frame


def _square(frame: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Center-crop ``frame`` to a square, then resize to (size, size) RGB."""
    from PIL import Image

    frame = _rgb(frame)
    h, w = frame.shape[0], frame.shape[1]
    edge = min(h, w)
    top = (h - edge) // 2
    left = (w - edge) // 2
    cropped = frame[top : top + edge, left : left + edge]

    if edge == size:
        return np.ascontiguousarray(cropped, dtype=np.uint8)
    img = Image.fromarray(cropped).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)

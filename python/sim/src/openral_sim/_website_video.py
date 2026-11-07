"""Clean single-view MP4 helper for website hero videos.

Unlike ``openral_sim._video`` (the two-band *debug* montage), this writes
*only* the world/viewer render — the frame an operator sees in the MuJoCo
viewer — with no VLA-input panel, no joint plot, and no burned-in overlays.

The website component (``website/``) is responsible for the rest of the
visuals: square crop (``object-fit: cover``), rounded corners, the
benchmark/rSkill labels, and the SUCCESS/FAIL status badge are all styled DOM
elements layered over the ``<video>``. The one burned-in element is a thin
outcome border: grey while the episode runs, then the colour of the placement
grade the scene reports (``EpisodeResult.frame_grades``, ``grade_colour``: green
success, yellow half credit, orange less) from the step it is reached, and the
episode's outcome (red on a fail) on the last frame, held for a second.

Frames keep their native aspect by default (a 4:3 top view is not cropped);
``square=True`` centre-crops to the square website canvas. ``size`` (the long
edge, or the square edge) controls only the written canvas; source sharpness
is bounded by the scene's native render resolution. We never touch the policy's
observation resolution, so recording does not change task outcomes.

Implementation notes
--------------------
* Uses imageio + imageio-ffmpeg for MP4 muxing (libx264, yuv420p) — same codec
  path as ``openral_sim._video``.
* With ``square=True`` each frame is center-cropped to a square, then resized
  to ``size × size`` (the component still cover-crops, harmlessly).
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
#: Grade colours (green success, yellow half credit, orange less, red fail) and the running grey.
_GREEN = (40, 200, 60)
_YELLOW = (240, 210, 30)
_ORANGE = (245, 130, 20)
_RED = (220, 40, 40)
_GREY = (128, 128, 128)


def grade_colour(score: float) -> tuple[int, int, int]:
    """RGB for a placement score: green 1.0, yellow >= 0.5, orange > 0, red 0.

    Shared by the video border and a scene's own grade overlay (the Metal marker ring), so
    both turn the same colour on the same frame.

    Example:
        >>> [grade_colour(s) for s in (1.0, 0.5, 0.25, 0.0)]
        [(40, 200, 60), (240, 210, 30), (245, 130, 20), (220, 40, 40)]
    """
    if score >= 1.0:
        return _GREEN
    if score >= 0.5:  # noqa: PLR2004 - restock's ``partial`` score
        return _YELLOW
    return _ORANGE if score > 0.0 else _RED


def _final_colour(r: EpisodeResult) -> tuple[int, int, int]:
    """Episode outcome: the success colour, else ``grade_colour`` of the episode's reward."""
    return _GREEN if r.success else grade_colour(r.total_reward)


def _frame_colour(r: EpisodeResult, i: int) -> tuple[int, int, int]:
    """Border of frame ``i``: the grade reached so far (grey while none), the outcome last.

    Indices past the last captured frame (the hold) take the outcome too.

    Example:
        >>> from openral_sim.rollout import EpisodeResult
        >>> r = EpisodeResult(frames=[0, 0, 0], frame_grades=[None, ("partial", 0.5), None])
        >>> [_frame_colour(r, i) for i in range(3)]
        [(128, 128, 128), (240, 210, 30), (220, 40, 40)]
    """
    if i >= len(r.frames) - 1:
        return _final_colour(r)
    g = r.frame_grades[i] if i < len(r.frame_grades) else None
    if g is None or g[1] <= 0.0:
        return _GREY
    return grade_colour(g[1])


def _outcome_label(r: EpisodeResult) -> str:
    """``partial 0.50`` for a graded episode, else ``success`` / ``reward 0.00``."""
    grade = r.final_info.get("placement_grade")
    if isinstance(grade, str):
        return f"{grade} {r.total_reward:.2f}"
    return "success" if r.success else f"reward {r.total_reward:.2f}"


def _bordered(
    frame: NDArray[np.uint8], colour: tuple[int, int, int], width: int
) -> NDArray[np.uint8]:
    """``frame`` with a ``width`` px ``colour`` outline (a copy)."""
    out = np.array(frame, dtype=np.uint8)
    out[:width], out[-width:], out[:, :width], out[:, -width:] = colour, colour, colour, colour
    return out


def save_world_mp4(
    result: EpisodeResult,
    path: Path,
    *,
    fps: int = 20,
    size: int = _DEFAULT_SIZE,
    min_duration_s: float = 2.0,
    square: bool = False,
) -> Path:
    """Write a clean world-view MP4 for one ``EpisodeResult``, with its outcome border.

    Args:
        result: An ``EpisodeResult`` produced by ``SimRunner`` with
            ``record_video=True``. Only ``frames`` (the rollout/world view) is
            used; ``vla_input_frames`` and ``joint_positions`` are ignored.
        path: Destination ``.mp4`` file. Parent directories are created.
        fps: Playback frame rate.
        size: Output long edge in pixels (the frame keeps its aspect), or the
            square edge with ``square``.
        min_duration_s: Minimum playback length. Short successful rollouts
            hold the final frame so website clips remain watchable. The final
            frame is always held at least one second, in the outcome colour.
        square: Center-crop each frame to a square (the website canvas) instead
            of keeping its native aspect.

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
        frames.extend([frames[-1]] * max(fps, min_frames - len(frames)))
        for i, frame in enumerate(frames):
            out = _square(frame, size) if square else _fit(frame, size)
            writer.append_data(_bordered(out, _frame_colour(result, i), max(2, size // 160)))
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
    the frame's edges is lost), carries an ``ep<i>`` label and its outcome (the grade
    name and score on a graded scene: ``ep3  partial 0.50``), and holds its last frame
    until the longest episode ends, plus one second. The outline is live: grey while
    the episode runs, the reported grade's colour from the step it is reached, and the
    outcome on its last frame (``grade_colour``: green success, yellow half credit,
    orange less, red otherwise).

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
    longest = max(len(r.frames) for _, r in clips) + fps  # + 1 s outcome hold
    border = max(2, tile_height // 60)

    def tile(t: int, i: int, r: EpisodeResult) -> NDArray[np.uint8]:
        frame = r.frames[min(t, len(r.frames) - 1)]
        img = Image.fromarray(_rgb(frame)).resize((tile_w, tile_height), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(img)
        draw.text((border + 4, border + 2), f"ep{i}  {_outcome_label(r)}", fill=(255, 255, 255))
        return _bordered(np.asarray(img, dtype=np.uint8), _frame_colour(r, t), border)

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
                y, x = (k // cols) * tile_height, (k % cols) * tile_w
                canvas[y : y + tile_height, x : x + tile_w] = tile(t, i, r)
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
    square: bool = False,
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
        size: Output long edge (square edge with ``square``) in pixels, passed to
            ``save_world_mp4``.
        fps: Playback frame rate.
        grid: Also write ``<scene>_<rskill>_grid.mp4``, every episode tiled in one
            clip (``write_world_grid``); not a website clip, so not in the manifest.
        square: Center-crop the per-episode clips to a square (the website canvas)
            instead of keeping the frame's native aspect.

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
        out = save_world_mp4(r, path, size=size, fps=fps, square=square)
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


def _fit(frame: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Resize ``frame`` (RGB, native aspect) so its long edge is ``size``; even sides."""
    from PIL import Image

    frame = _rgb(frame)
    h, w = frame.shape[0], frame.shape[1]
    scale = size / max(h, w)
    out_w = max(_MIN_EVEN_SIDE, round(w * scale / 2) * 2)
    out_h = max(_MIN_EVEN_SIDE, round(h * scale / 2) * 2)
    if (out_h, out_w) == (h, w):
        return np.ascontiguousarray(frame, dtype=np.uint8)
    img = Image.fromarray(frame).resize((out_w, out_h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


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

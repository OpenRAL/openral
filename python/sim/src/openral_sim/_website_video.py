"""Clean single-view MP4 helper for website hero videos.

Unlike :mod:`openral_sim._video` (the 3-panel *debug* montage), this writes
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
  path as :mod:`openral_sim._video`.
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


def save_world_mp4(
    result: EpisodeResult,
    path: Path,
    *,
    fps: int = 20,
    size: int = _DEFAULT_SIZE,
    min_duration_s: float = 2.0,
) -> Path:
    """Write a clean square world-view MP4 for one :class:`EpisodeResult`.

    Args:
        result: An :class:`EpisodeResult` produced by :class:`SimRunner` with
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


def write_world_videos(
    episodes: list[EpisodeResult],
    out_dir: Path,
    *,
    scene: str,
    rskill: str,
    section: str,
    size: int = _DEFAULT_SIZE,
    fps: int = 20,
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
        size: Square output edge in pixels, passed to :func:`save_world_mp4`.
        fps: Playback frame rate.

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


def _slug(s: str) -> str:
    """Flatten an id into a single filename token (no path separators)."""
    return s.replace("/", "_").replace("\\", "_")


def _square(frame: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Center-crop ``frame`` to a square, then resize to (size, size) RGB."""
    from PIL import Image

    if frame.ndim == _GRAYSCALE_NDIM:
        frame = np.stack([frame] * _RGB_CHANNELS, axis=-1)
    elif frame.shape[2] != _RGB_CHANNELS:
        frame = np.repeat(frame[:, :, :1], _RGB_CHANNELS, axis=2)

    h, w = frame.shape[0], frame.shape[1]
    edge = min(h, w)
    top = (h - edge) // 2
    left = (w - edge) // 2
    cropped = frame[top : top + edge, left : left + edge]

    if edge == size:
        return np.ascontiguousarray(cropped, dtype=np.uint8)
    img = Image.fromarray(cropped).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)

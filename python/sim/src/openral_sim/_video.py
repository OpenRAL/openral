"""Shared MP4 helper for the runnable eval examples.

Every example writes the same debug video:

    +-----------------------------------------------+
    |   VLA input view / policy camera grid         |
    |   (post-preprocess image(s) the policy saw)   |
    +-----------------------------------------------+
    |    observation state over time (line plot)    |
    +-----------------------------------------------+

This is the one helper every example must use — there is no
example-specific video processing. Differences in the underlying VLA
(SmolVLA's wrist + agent-view cameras, π0.5's PaliGemma view, ACT's
``top`` camera, …) all flow through ``EpisodeResult.vla_input_frames``,
``EpisodeResult.frames``, and ``EpisodeResult.joint_positions`` — which
the strict runner populates uniformly.

Implementation notes
--------------------
* Uses imageio + imageio-ffmpeg for MP4 muxing (libx264, yuv420p).
* State plot is rendered once with a headless matplotlib figure; each
  frame copies that raster and paints the *moving cursor* (the time axis
  position that follows the rollout) with numpy, so a frame costs no
  matplotlib redraw.
* The video is written at most at ``_MAX_VIDEO_FPS``: a faster control rate
  (a 60 Hz robot) is sampled down, so playback stays real-time while every
  per-frame cost (resize, annotate, encode) drops by the same factor. Panels
  are built per written frame, and a source frame that is the same object as
  the previous one (a policy preview held between inferences) is reused.
* If ``vla_input_frames`` is empty (mock / scripted policies that don't
  consume images), the top panel falls back to the rollout/world stream.
* If ``joint_positions`` is empty (env doesn't expose ``state``), the
  bottom panel is replaced with a "no proprioception" placeholder.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from openral_sim.rollout import EpisodeResult


# Output canvas dimensions. Top row is two square tiles side-by-side;
# bottom row is the joint plot, sized so the overall canvas is 16:9-ish.
_TILE_SIZE = 256  # square tile for each top-row video
_TOP_W = _TILE_SIZE * 2
_PLOT_H = 192
_CANVAS_W = _TOP_W
_CANVAS_H = _TILE_SIZE + _PLOT_H

_RGB_CHANNELS = 3
_GRAYSCALE_NDIM = 2
# How many joints can fit in the legend before we drop labels (otherwise
# the legend dominates the plot).
_MAX_LEGEND_JOINTS = 8
_LABEL_PAD = 6
_LABEL_BAR_H = 24
# Highest frame rate a debug video is written at (see module docstring).
_MAX_VIDEO_FPS = 30
# Cursor colour and opacity (matplotlib "red", alpha 0.7) and half-width in pixels.
_CURSOR_RGB = np.array([255, 0, 0], dtype=np.float32)
_CURSOR_ALPHA = 0.7
_CURSOR_HALF_W = 1


def save_episode_mp4(
    result: EpisodeResult,
    path: Path,
    *,
    fps: int = 20,
    title: str | None = None,
) -> Path:
    """Write a two-band debug MP4 for one ``EpisodeResult``.

    Args:
        result: An ``EpisodeResult`` produced by ``SimRunner`` with
            ``record_video=True``. ``vla_input_frames`` populates the top
            panel when present; otherwise ``frames`` (rollout/world) is used
            as the fallback. ``joint_positions`` populates the bottom plot.
        path: Destination ``.mp4`` file. Parent directories are created.
        fps: The rate the steps were recorded at (playback is real-time); the
            file is written at ``min(fps, _MAX_VIDEO_FPS)``, sampling steps.
        title: Optional title text rendered above the joint plot.

    Returns:
        The output path.

    Raises:
        ValueError: If both ``frames`` and ``vla_input_frames`` are empty
            (no video to render) — set ``record_video=True`` first; or if
            ``path`` does not have an ``.mp4`` suffix (this helper is
            ffmpeg/x264/yuv420p only — see module docstring).
    """
    if not result.frames and not result.vla_input_frames:
        raise ValueError(
            "EpisodeResult has no frames; set SimEnvironment.record_video=True "
            "before driving SimRunner to capture rollout + VLA-input frames"
        )
    if path.suffix.lower() != ".mp4":
        raise ValueError(
            f"save_episode_mp4 only writes MP4 (libx264/yuv420p); got path={path!s} "
            f"with suffix {path.suffix!r}. Use a '.mp4' extension — the helper is "
            "intentionally MP4-only (see module docstring)."
        )

    # Length-align the per-step series. ``frames`` and ``vla_input_frames``
    # may be 1 longer than ``actions`` because the runner records the frame
    # *before* policy.step. We trim to the shortest non-empty length to keep
    # the joint cursor in lockstep with the videos.
    n_rollout = len(result.frames)
    n_vla = len(result.vla_input_frames)
    n_state = len(result.joint_positions)
    n = max(n_rollout, n_vla, n_state, 1)

    # Multi-cam VLAs already encode their camera set into the stitched
    # ``vla_input_frames`` preview. Show that grid full-width; the rollout
    # render is often redundant or misleadingly similar to one of the cameras.
    # For policies without input frames, fall back to the rollout/world view.
    show_two = result.num_input_cameras > 1 and bool(result.vla_input_frames)

    states = _stack_padded_states(result.joint_positions, n)
    # Multi-cam: the stitched policy-input grid; else whichever stream is
    # non-empty, preferring the rollout view.
    source = result.vla_input_frames if show_two else (result.frames or result.vla_input_frames)
    label = (
        f"policy inputs ({result.num_input_cameras} cams)" if show_two else "rollout / policy view"
    )
    out_fps = min(fps, _MAX_VIDEO_FPS)
    steps = _sampled_steps(n, fps, out_fps)

    plot_renderer = _JointPlotRenderer(
        states=states,
        width=_CANVAS_W,
        height=_PLOT_H,
        title=title,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as iio

    writer = iio.get_writer(
        str(path),
        format="ffmpeg",  # type: ignore[arg-type]  # reason: imageio v2 stubs type format as Format enum but accept plugin strings at runtime
        fps=out_fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,  # no resize; we build exact-size canvases
        pixelformat="yuv420p",
    )
    try:
        last_src: NDArray[np.uint8] | None = None
        top: NDArray[np.uint8] = np.zeros((_TILE_SIZE, _TOP_W, 3), dtype=np.uint8)
        for i in steps:
            src = _frame_at(source, i, n)
            if src is None or src is not last_src:
                panel = (
                    _resize_frame(src, _TOP_W, _TILE_SIZE)
                    if src is not None
                    else np.zeros((_TILE_SIZE, _TOP_W, 3), dtype=np.uint8)
                )
                top = _annotate_panel(panel, label)
                last_src = src
            plot = plot_renderer.render_at_step(i)  # (PLOT_H, CANVAS_W, 3)
            canvas = np.concatenate([top, plot], axis=0)  # (CANVAS_H, CANVAS_W, 3)
            writer.append_data(canvas)
    finally:
        writer.close()
    return path


# ────────────────────────── helpers ──────────────────────────


def _stack_padded_states(
    states: list[NDArray[np.float32]],
    target_len: int,
) -> NDArray[np.float32]:
    """Stack ragged per-step state vectors into a (T, D) array, padding short ones."""
    if not states:
        return np.zeros((target_len, 1), dtype=np.float32)
    dim = max(s.shape[0] for s in states)
    arr = np.zeros((target_len, dim), dtype=np.float32)
    for i in range(target_len):
        s = states[min(i, len(states) - 1)]
        arr[i, : s.shape[0]] = s
    return arr


def _sampled_steps(n: int, fps: int, out_fps: int) -> list[int]:
    """The step shown by each written frame when ``fps`` steps/s play at ``out_fps``.

    Example:
        >>> _sampled_steps(6, 60, 30)
        [0, 2, 4]
        >>> _sampled_steps(3, 15, 30)
        [0, 1, 2]
    """
    if out_fps >= fps:
        return list(range(n))
    count = max(1, -(-n * out_fps // fps))  # ceil
    return [min(n - 1, round(k * fps / out_fps)) for k in range(count)]


def _frame_at(frames: list[NDArray[np.uint8]], step: int, n_steps: int) -> NDArray[np.uint8] | None:
    """The frame for ``step`` of ``n_steps``: sampled uniformly, the last held; ``None`` if empty.

    Example:
        >>> f = [np.zeros((1, 1, 3), np.uint8), np.ones((1, 1, 3), np.uint8)]
        >>> [int(_frame_at(f, s, 4)[0, 0, 0]) for s in range(4)]
        [0, 0, 1, 1]
    """
    if not frames:
        return None
    return frames[min(int(step * len(frames) / max(n_steps, 1)), len(frames) - 1)]


def _annotate_panel(frame: NDArray[np.uint8], label: str) -> NDArray[np.uint8]:
    """Add a small text banner to a video panel without changing its shape."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle(
        (0, 0, frame.shape[1], _LABEL_BAR_H),
        fill=(20, 20, 20, 190),
    )
    draw.text((_LABEL_PAD, 4), label, fill=(255, 255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def _resize_frame(
    frame: NDArray[np.uint8],
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    """Resize ``frame`` to (height, width) using PIL nearest-neighbour."""
    from PIL import Image

    if frame.shape[2] != _RGB_CHANNELS:
        # Grayscale → broadcast to RGB.
        if frame.ndim == _GRAYSCALE_NDIM:
            frame = np.stack([frame] * _RGB_CHANNELS, axis=-1)
        else:
            frame = np.repeat(frame[:, :, :1], _RGB_CHANNELS, axis=2)
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    img = Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class _JointPlotRenderer:
    """Renders the bottom observation-state panel with a moving time cursor.

    Draws the static line plot once, then per frame copies that raster and
    paints the vertical cursor with numpy at the step's pixel column (from the
    axes' data transform), blended like the matplotlib line it replaces. No
    matplotlib draw per frame.
    """

    def __init__(
        self,
        states: NDArray[np.float32],
        *,
        width: int,
        height: int,
        title: str | None,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self._width = width
        self._height = height
        self._states = states  # (T, D)
        self._n = states.shape[0]
        self._dim = states.shape[1] if states.size else 0

        # Build figure at exact pixel size (dpi=100 → 1 inch = 100 px).
        dpi = 100
        fig, ax = plt.subplots(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
        )
        if title is not None:
            ax.set_title(title, fontsize=9)

        if self._dim == 0 or self._n == 0:
            ax.text(
                0.5,
                0.5,
                "no proprioception captured",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="0.3",
            )
        else:
            t = np.arange(self._n)
            for d in range(self._dim):
                label = f"s{d}" if self._dim <= _MAX_LEGEND_JOINTS else None
                ax.plot(t, states[:, d], linewidth=1.0, label=label)
            if self._dim <= _MAX_LEGEND_JOINTS:
                ax.legend(loc="upper right", fontsize=7, ncol=min(self._dim, 4))
            ax.set_xlim(0, max(self._n - 1, 1))
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("observation state", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout(pad=0.5)
        self._fig = fig
        self._ax = ax
        self._plt = plt
        # Cache the static background once, and where the cursor may go: the axes'
        # pixel rows, and each step's pixel column (display y runs bottom-up).
        self._fig.canvas.draw()
        self._static_rgb = self._snapshot()
        bbox = ax.get_window_extent()
        h = self._static_rgb.shape[0]
        self._rows = slice(max(0, round(h - bbox.y1)), min(h, round(h - bbox.y0)))
        self._x_of = ax.transData

    def render_at_step(self, step: int) -> NDArray[np.uint8]:
        if self._n == 0:
            return self._static_rgb
        # Agg snaps a 1-px line to the pixel left of the transformed centre.
        x = int(float(self._x_of.transform((step, 0.0))[0]) - 0.5)
        cols = slice(max(0, x - _CURSOR_HALF_W), min(self._width, x + _CURSOR_HALF_W + 1))
        out = self._static_rgb.copy()
        band = out[self._rows, cols].astype(np.float32)
        out[self._rows, cols] = (band * (1.0 - _CURSOR_ALPHA) + _CURSOR_RGB * _CURSOR_ALPHA).astype(
            np.uint8
        )
        return out

    def _snapshot(self) -> NDArray[np.uint8]:
        # Pull the rendered canvas into an RGB array sized to (H, W).
        fig = self._fig
        canvas = fig.canvas
        # canvas.buffer_rgba returns a memoryview; reshape to (h, w, 4).
        # Only Agg/Cairo canvases expose buffer_rgba; we configure the
        # figure with `Agg` above so the cast is safe but mypy can't see it.
        buf = np.asarray(canvas.buffer_rgba())  # type: ignore[attr-defined]  # reason: matplotlib base FigureCanvas has no buffer_rgba; Agg/Cairo concrete subclasses do
        rgb = buf[..., :3]
        if rgb.shape[0] == self._height and rgb.shape[1] == self._width:
            return rgb.copy()
        return _resize_frame(rgb, self._width, self._height)

    def __del__(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._plt.close(self._fig)

"""Shared MP4 helper for the runnable eval examples.

Every example writes the same debug video:

    +-----------------------------------------------+
    |   VLA input view / policy camera grid         |
    |   (post-preprocess image(s) the policy saw)   |
    +-----------------------------------------------+
    |    joint positions over time (line plot)      |
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
* Joint plot is rendered with a headless matplotlib figure once per
  frame and rasterised into the canvas; this is slower than a static
  plot but yields a *moving cursor* on the time axis that follows the
  rollout, which is what makes the video useful for debugging.
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


def save_episode_mp4(
    result: EpisodeResult,
    path: Path,
    *,
    fps: int = 20,
    title: str | None = None,
) -> Path:
    """Write a 3-panel debug MP4 for one :class:`EpisodeResult`.

    Args:
        result: An :class:`EpisodeResult` produced by :class:`SimRunner` with
            ``record_video=True``. ``vla_input_frames`` populates the top
            panel when present; otherwise ``frames`` (rollout/world) is used
            as the fallback. ``joint_positions`` populates the bottom plot.
        path: Destination ``.mp4`` file. Parent directories are created.
        fps: Playback frame rate.
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
    if show_two:
        top_w = _TILE_SIZE * 2
        top_seq = _resize_sequence(result.vla_input_frames, top_w, _TILE_SIZE, target_len=n)
    else:
        top_w = _TILE_SIZE * 2
        # Pick whichever stream is non-empty; prefer the rollout view.
        primary = result.frames or result.vla_input_frames
        top_seq = _resize_sequence(primary, top_w, _TILE_SIZE, target_len=n)

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
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,  # no resize; we build exact-size canvases
        pixelformat="yuv420p",
    )
    try:
        for i in range(n):
            if show_two:
                top = _annotate_panel(top_seq[i], f"policy inputs ({result.num_input_cameras} cams)")
            else:
                top = _annotate_panel(
                    top_seq[i],
                    "rollout / policy view",
                )  # already (H, 2W, 3) when single-panel
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


def _resize_sequence(
    frames: list[NDArray[np.uint8]],
    width: int,
    height: int,
    *,
    target_len: int,
) -> list[NDArray[np.uint8]]:
    """Resize a sequence of HWC uint8 frames to (height, width). Black-fill if empty.

    If ``frames`` is shorter than ``target_len``, the last frame is held;
    if longer, it is sampled uniformly.
    """
    if not frames:
        black = np.zeros((height, width, 3), dtype=np.uint8)
        return [black] * target_len
    out: list[NDArray[np.uint8]] = []
    n = len(frames)
    for i in range(target_len):
        # Map step i ∈ [0, target_len) → frame index ∈ [0, n).
        idx = min(int(i * n / max(target_len, 1)), n - 1)
        out.append(_resize_frame(frames[idx], width, height))
    return out


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
    """Renders the bottom joint-positions panel with a moving time cursor.

    Draws the static line plot once and only updates the vertical cursor
    per frame — keeps the per-step cost cheap (~1 ms vs ~30 ms for a full
    redraw).
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
                label = f"j{d}" if self._dim <= _MAX_LEGEND_JOINTS else None
                ax.plot(t, states[:, d], linewidth=1.0, label=label)
            if self._dim <= _MAX_LEGEND_JOINTS:
                ax.legend(loc="upper right", fontsize=7, ncol=min(self._dim, 4))
            ax.set_xlim(0, max(self._n - 1, 1))
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("joint pos", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)

        # Vertical cursor we'll move per-frame.
        self._cursor = ax.axvline(0, color="red", linewidth=1.2, alpha=0.7)
        fig.tight_layout(pad=0.5)
        self._fig = fig
        self._ax = ax
        self._plt = plt
        # Cache the static background once.
        self._fig.canvas.draw()
        self._static_rgb = self._snapshot()

    def render_at_step(self, step: int) -> NDArray[np.uint8]:
        if self._n == 0:
            return self._static_rgb
        self._cursor.set_xdata([step])
        self._fig.canvas.draw()
        return self._snapshot()

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
